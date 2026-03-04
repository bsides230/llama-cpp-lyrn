import os
import ctypes
from typing import List, Dict, Any, Optional
import weakref

from . import llama_cpp

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
        add_bos: bool = True,
        add_eos: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.n_batch = n_batch
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

    def create_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        if not stream:
            raise ValueError("Only stream=True is supported by this minimal binding")

        # FIX: Keep references to the bytes objects so they are not garbage collected
        # while the C code is accessing the pointers in c_messages array!
        encoded_roles = []
        encoded_contents = []

        c_messages = (llama_cpp.llama_chat_message * len(messages))()
        for i, msg in enumerate(messages):
            r = msg["role"].encode("utf-8")
            c = msg["content"].encode("utf-8")
            encoded_roles.append(r)
            encoded_contents.append(c)

            c_messages[i].role = r
            c_messages[i].content = c

        tmpl = llama_cpp._lib.llama_model_chat_template(self.model, None)

        buf = (ctypes.c_char * 4096)()
        n = llama_cpp._lib.llama_chat_apply_template(
            tmpl,
            c_messages,
            len(messages),
            True,
            buf,
            len(buf)
        )
        if n < 0:
            buf = (ctypes.c_char * -n)()
            n = llama_cpp._lib.llama_chat_apply_template(
                tmpl,
                c_messages,
                len(messages),
                True,
                buf,
                len(buf)
            )

        # If template application still fails, fall back to NULL (chatml default)
        if n < 0:
            buf = (ctypes.c_char * 4096)()
            n = llama_cpp._lib.llama_chat_apply_template(
                None,
                c_messages,
                len(messages),
                True,
                buf,
                len(buf)
            )
            if n < 0:
                raise RuntimeError(f"llama_chat_apply_template failed with error code {n}")

        prompt = buf.raw[:n].decode("utf-8")
        tokens = self._tokenize(prompt, add_special=False, parse_special=True)

        if not tokens:
            raise RuntimeError("Tokenization produced no tokens from the prompt")

        chain_params = llama_cpp._lib.llama_sampler_chain_default_params()
        chain = llama_cpp._lib.llama_sampler_chain_init(chain_params)

        top_k = kwargs.get("top_k", 40)
        top_p = kwargs.get("top_p", 0.95)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_top_k(top_k))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_top_p(top_p, 1))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_temp(temperature))
        llama_cpp._lib.llama_sampler_chain_add(chain, llama_cpp._lib.llama_sampler_init_dist(42))

        return self._generate_stream(tokens, chain, max_tokens)

    def _generate_stream(self, prompt_tokens: List[int], sampler: llama_cpp.llama_sampler_p, max_tokens: int):
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

        while n_cur <= total_tokens + max_tokens:
            new_token_id = llama_cpp._lib.llama_sampler_sample(sampler, self.ctx, -1)
            llama_cpp._lib.llama_sampler_accept(sampler, new_token_id)

            if new_token_id == eos_token:
                break

            piece = self._token_to_piece(new_token_id)
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": piece
                        }
                    }
                ]
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
