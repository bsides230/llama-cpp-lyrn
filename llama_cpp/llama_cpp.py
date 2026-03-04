import ctypes
import os
import sys

# Load shared library
_lib_name = "libllama.so"
if sys.platform == "win32":
    _lib_name = "llama.dll"
elif sys.platform == "darwin":
    _lib_name = "libllama.dylib"

_lib_path = os.path.join(os.path.dirname(__file__), "lib", _lib_name)
try:
    _lib = ctypes.CDLL(_lib_path)
except OSError as e:
    raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p
llama_vocab_p = ctypes.c_void_p
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("rpc_servers", ctypes.c_char_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]

_lib.llama_backend_init.argtypes = []
_lib.llama_backend_init.restype = None

_lib.llama_backend_free.argtypes = []
_lib.llama_backend_free.restype = None

_lib.llama_model_default_params.argtypes = []
_lib.llama_model_default_params.restype = llama_model_params

_lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
_lib.llama_model_load_from_file.restype = llama_model_p

_lib.llama_model_free.argtypes = [llama_model_p]
_lib.llama_model_free.restype = None

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
    ]

_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params

_lib.llama_init_from_model.argtypes = [llama_model_p, llama_context_params]
_lib.llama_init_from_model.restype = llama_context_p

_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

_lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_lib.llama_batch_init.restype = llama_batch

_lib.llama_batch_free.argtypes = [llama_batch]
_lib.llama_batch_free.restype = None

_lib.llama_model_get_vocab.argtypes = [llama_model_p]
_lib.llama_model_get_vocab.restype = llama_vocab_p

_lib.llama_vocab_eos.argtypes = [llama_vocab_p]
_lib.llama_vocab_eos.restype = llama_token

_lib.llama_tokenize.argtypes = [llama_vocab_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
_lib.llama_tokenize.restype = ctypes.c_int32

_lib.llama_token_to_piece.argtypes = [llama_vocab_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
_lib.llama_token_to_piece.restype = ctypes.c_int32

_lib.llama_decode.argtypes = [llama_context_p, llama_batch]
_lib.llama_decode.restype = ctypes.c_int32

class llama_sampler_p(ctypes.c_void_p):
    pass

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]

_lib.llama_sampler_chain_default_params.argtypes = []
_lib.llama_sampler_chain_default_params.restype = llama_sampler_chain_params

_lib.llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
_lib.llama_sampler_chain_init.restype = llama_sampler_p

_lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
_lib.llama_sampler_init_top_p.restype = llama_sampler_p

_lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
_lib.llama_sampler_init_top_k.restype = llama_sampler_p

_lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
_lib.llama_sampler_init_temp.restype = llama_sampler_p

_lib.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
_lib.llama_sampler_chain_add.restype = None

_lib.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, ctypes.c_int32]
_lib.llama_sampler_sample.restype = llama_token

_lib.llama_sampler_accept.argtypes = [llama_sampler_p, llama_token]
_lib.llama_sampler_accept.restype = None

_lib.llama_sampler_free.argtypes = [llama_sampler_p]
_lib.llama_sampler_free.restype = None

class llama_chat_message(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]

_lib.llama_chat_apply_template.argtypes = [
    llama_model_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool, ctypes.c_char_p, ctypes.c_int32
]
_lib.llama_chat_apply_template.restype = ctypes.c_int32
