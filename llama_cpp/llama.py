import os
import ctypes
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
import weakref
import sys
import numpy as np
import numpy.typing as npt

from . import llama_cpp
from . import llama_chat_format
from . import llama_types

LogitsProcessor = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], npt.NDArray[np.single]]

class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores

StoppingCriteria = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], bool]

class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
    ) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])

class Llama:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        n_batch: int = 512,
        use_mlock: bool = False,
        use_mmap: bool = True,
        chat_format: Optional[str] = None,
        chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.n_batch = n_batch
        self.chat_format = chat_format
        self.chat_handler = chat_handler
        self._chat_handlers: Dict[str, llama_chat_format.LlamaChatCompletionHandler] = {}

        # Pre-register available chat formats
        template_choices = {} # Could be populated from GGUF metadata
        for name, template in template_choices.items():
            self._chat_handlers[name] = llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=self.token_eos(),
                bos_token=self.token_bos(),
                stop_token_ids=[self.token_eos()],
            ).to_chat_handler()

        if self.chat_format is None and self.chat_handler is None:
            self.chat_format = "llama-2"
            if self.verbose:
                print(f"Using fallback chat format: {self.chat_format}", file=sys.stderr)
        if self.verbose:
            print(f"llama-cpp-lyrn: Loading model from {model_path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        llama_cpp._lib.llama_backend_init()

        model_params = llama_cpp._lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        model_params.use_mlock = use_mlock
        model_params.use_mmap = use_mmap

        self.model = llama_cpp._lib.llama_model_load_from_file(model_path.encode("utf-8"), model_params)
        if not self.model:
            raise RuntimeError(f"Failed to load model from {model_path}")

        ctx_params = llama_cpp._lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads

        self.ctx = llama_cpp._lib.llama_init_from_model(self.model, ctx_params)
        if not self.ctx:
            raise RuntimeError("Failed to create llama_context")

        # Create reusable batch
        self.batch = llama_cpp._lib.llama_batch_init(n_batch, 0, 1)

        # Validate batch pointers
        if not self.batch.token:
            raise RuntimeError("llama_batch_init returned batch with NULL token pointer")

        self.n_past = 0

        # Setup finalizer properly
        self._finalizer = weakref.finalize(self, self._cleanup_resources, llama_cpp._lib, self.model, self.ctx, self.batch)

    @staticmethod
    def _cleanup_resources(_lib, model, ctx, batch):
        if batch is not None:
            _lib.llama_batch_free(batch)
        if ctx is not None:
            _lib.llama_free(ctx)
        if model is not None:
            _lib.llama_model_free(model)

    def close(self):
        if hasattr(self, "_finalizer"):
            self._finalizer()

    def token_bos(self) -> int:
        vocab = llama_cpp._lib.llama_model_get_vocab(self.model)
        return llama_cpp._lib.llama_vocab_bos(vocab)

    def token_eos(self) -> int:
        vocab = llama_cpp._lib.llama_model_get_vocab(self.model)
        return llama_cpp._lib.llama_vocab_eos(vocab)

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        text_str = text.decode("utf-8", errors="ignore")
        return self._tokenize(text_str, add_special=add_bos, parse_special=special)

    def detokenize(self, tokens: List[int]) -> bytes:
        result = ""
        for token in tokens:
            result += self._token_to_piece(token)
        return result.encode("utf-8")

    def reset(self):
        """Reset the model state."""
        self.n_past = 0

    def _tokenize(self, text: str, add_special: bool, parse_special: bool = False) -> List[int]:
        text_bytes = text.encode("utf-8")
        # Allocate at least 1 token to avoid zero-length array issues
        n_tokens = max(len(text_bytes) + (1 if add_special else 0), 1)
        tokens = (llama_cpp.llama_token * n_tokens)()
        vocab = llama_cpp._lib.llama_model_get_vocab(self.model)

        n = llama_cpp._lib.llama_tokenize(
            vocab,
            text_bytes,
            len(text_bytes),
            tokens,
            n_tokens,
            add_special,
            parse_special
        )
        if n < 0:
            tokens = (llama_cpp.llama_token * -n)()
            n = llama_cpp._lib.llama_tokenize(
                vocab,
                text_bytes,
                len(text_bytes),
                tokens,
                -n,
                add_special,
                parse_special
            )

        return [tokens[i] for i in range(n)]

    def _token_to_piece(self, token: int) -> str:
        buf = (ctypes.c_char * 32)()
        vocab = llama_cpp._lib.llama_model_get_vocab(self.model)
        n = llama_cpp._lib.llama_token_to_piece(vocab, token, buf, len(buf), 0, False)
        if n < 0:
            buf = (ctypes.c_char * -n)()
            n = llama_cpp._lib.llama_token_to_piece(vocab, token, buf, len(buf), 0, False)

        return buf.raw[:n].decode("utf-8", errors="ignore")

    def create_chat_completion(
        self,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[Any] = None,
        grammar: Optional[Any] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        handler = (
            self.chat_handler
            or self._chat_handlers.get(self.chat_format)
            or llama_chat_format.get_chat_completion_handler(self.chat_format)
        )
        return handler(
            llama=self,
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
            **kwargs,
        )

    def create_completion(self, prompt: Union[str, List[int]], stream: bool = False, **kwargs) -> Any:
        tokens = prompt
        if isinstance(prompt, str):
            tokens = self._tokenize(prompt, add_special=False, parse_special=True)
            if not tokens:
                raise RuntimeError("Tokenization produced no tokens from the prompt")

        chain_params = llama_cpp._lib.llama_sampler_chain_default_params()
        chain = llama_cpp._lib.llama_sampler_chain_init(chain_params)

        top_k = kwargs.get("top_k", 40)
        top_p = kwargs.get("top_p", 0.95)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)
        grammar = kwargs.get("grammar", None)

        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_top_k(top_k))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_top_p(top_p, 1))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_temp(temperature))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_dist(42))

        if grammar is not None and hasattr(grammar, "_grammar") and grammar._grammar is not None:
            vocab = llama_cpp._lib.llama_model_get_vocab(self.model)
            llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_grammar(vocab, grammar._grammar.encode('utf-8'), grammar._root.encode('utf-8')))

        iterator = self._generate_stream(tokens, chain, max_tokens, stop=kwargs.get("stop", []))
        if stream:
            return iterator

        import time
        created = int(time.time())
        text = ""
        for chunk in iterator:
            text += chunk["choices"][0]["text"]

        return {
            "id": "cmpl-xyz",
            "object": "text_completion",
            "created": created,
            "model": "model",
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(text),
                "total_tokens": len(tokens) + len(text)
            }
        }

    def _generate_stream(self, prompt_tokens: List[int], sampler: llama_cpp.llama_sampler_p, max_tokens: int, stop: Optional[Union[str, List[str]]] = None):
        vocab = llama_cpp._lib.llama_model_get_vocab(self.model)
        eos_token = llama_cpp._lib.llama_vocab_eos(vocab)

        # Process prompt in batches if it exceeds batch capacity
        batch_size = self.n_batch
        total_tokens = len(prompt_tokens)

        n_cur = 0
        for batch_start in range(0, total_tokens, batch_size):
            batch_end = min(batch_start + batch_size, total_tokens)
            chunk = prompt_tokens[batch_start:batch_end]

            self.batch.n_tokens = 0
            for i, tok in enumerate(chunk):
                self.batch.token[i] = tok
                self.batch.pos[i] = self.n_past + i
                self.batch.n_seq_id[i] = 1
                self.batch.seq_id[i][0] = 0
                # Only request logits for the very last token of the entire prompt
                self.batch.logits[i] = 1 if (batch_start + i == total_tokens - 1) else 0
            self.batch.n_tokens = len(chunk)

            ret = llama_cpp._lib.llama_decode(self.ctx, self.batch)
            if ret != 0:
                llama_cpp._lib.llama_sampler_free(sampler)
                raise RuntimeError(f"llama_decode failed with error code {ret}")

            self.n_past += len(chunk)

        n_cur = total_tokens

        stop_list = []
        if isinstance(stop, str):
            stop_list = [stop]
        elif isinstance(stop, list):
            stop_list = stop

        generated_text = ""

        # Dummy id and created time for compatibility
        chunk_id = "cmpl-xyz"
        created = 0

        # yield first chunk with no text
        yield {
            "id": chunk_id,
            "object": "text_completion",
            "created": created,
            "model": "model",
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }

        while n_cur <= total_tokens + max_tokens:
            new_token_id = llama_cpp._lib.llama_sampler_sample(sampler, self.ctx, -1)
            llama_cpp._lib.llama_sampler_accept(sampler, new_token_id)

            if new_token_id == eos_token:
                break

            piece = self._token_to_piece(new_token_id)
            generated_text += piece

            should_stop = False
            for s in stop_list:
                if generated_text.endswith(s):
                    should_stop = True
                    # Truncate the stop string from the final piece if we yielded it
                    piece = piece[:len(piece)-len(s)]
                    break

            if should_stop:
                if piece:
                    yield {
                        "id": chunk_id,
                        "object": "text_completion",
                        "created": created,
                        "model": "model",
                        "choices": [
                            {
                                "text": piece,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                break

            yield {
                "id": chunk_id,
                "object": "text_completion",
                "created": created,
                "model": "model",
                "choices": [
                    {
                        "text": piece,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }

            self.batch.n_tokens = 0
            self.batch.token[0] = new_token_id
            self.batch.pos[0] = self.n_past
            self.batch.n_seq_id[0] = 1
            self.batch.seq_id[0][0] = 0
            self.batch.logits[0] = 1
            self.batch.n_tokens = 1

            ret = llama_cpp._lib.llama_decode(self.ctx, self.batch)
            if ret != 0:
                llama_cpp._lib.llama_sampler_free(sampler)
                raise RuntimeError(f"llama_decode failed during generation with error code {ret}")

            self.n_past += 1
            n_cur += 1

        llama_cpp._lib.llama_sampler_free(sampler)
