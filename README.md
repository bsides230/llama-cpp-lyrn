# LYRN Minimal Llama Binding

This repository is a **minimal, deterministic Python binding for
llama.cpp**, tailored specifically for LYRN.

This is NOT a general-purpose wrapper.\
This is NOT a full fork of llama-cpp-python.\
This is NOT intended to support every llama.cpp feature.

This repository exists to provide:

-   Stable embedded inference
-   Full control over context lifecycle
-   Zero build fragility from upstream tools
-   Support for newest GGUF architectures (including `qwen35`)
-   CPU-only edge deployment (Termux / Linux nodes)

------------------------------------------------------------------------

## 🎯 Design Goals

1.  **Minimal Surface Area**
    -   Keep only what LYRN uses.
    -   Delete everything else.
2.  **Library-Only Build**
    -   llama.cpp must build as a dependency.
    -   Do NOT build:
        -   tools/
        -   tools/mtmd
        -   examples/
        -   server/
        -   tests/
        -   benchmarks/
3.  **Text-Only**
    -   Multimodal is out of scope.
    -   mtmd must never enter the build graph.
4.  **Deterministic Context Control**
    -   LYRN manually controls when context resets.
    -   Binding must not hide context lifecycle.
5.  **Newest Qwen 3.5 Support**
    -   Vendor `ggml-org/llama.cpp`
    -   Must load GGUF with: `general.architecture = qwen35`
    -   No "unknown model architecture" errors allowed.

------------------------------------------------------------------------

## 📦 Required Public API

The binding must expose:

``` python
from llama_cpp import Llama
```

### Constructor

``` python
Llama(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
    n_batch: int,
    use_mlock: bool,
    use_mmap: bool,
    chat_format: Optional[str],
    add_bos: bool,
    add_eos: bool,
    verbose: bool,
)
```

Extra kwargs must be accepted safely (ignored if unused).

------------------------------------------------------------------------

### Required Method

``` python
create_chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stream: bool
)
```

If `stream=True`, this must yield incremental chunks shaped exactly
like:

``` python
{
    "choices": [
        {
            "delta": {
                "content": "..."
            }
        }
    ]
}
```

This contract must never change.

------------------------------------------------------------------------

## 🧠 Context Handling Rules

-   Model stays loaded in memory.
-   Context persists across calls unless explicitly reset.
-   Binding must expose a clean internal reset mechanism.
-   No implicit resets.

LYRN handles: - Snapshot rebuilding - Manual context wipes - Stop
triggers

The binding must not interfere.

------------------------------------------------------------------------

## 🔧 Build Rules

When building as a Python extension:

-   Must pass: `-DGGML_VULKAN=OFF`
-   Must pass: `-DLLAMA_DEPENDENCY_BUILD=ON` (or equivalent mechanism)

The build MUST:

-   Never configure tools/mtmd
-   Never build server
-   Never build examples
-   Never build tests

Build must succeed in Termux with:

``` bash
pip install --no-binary :all: .
```

------------------------------------------------------------------------

## 📁 Vendor Policy

`vendor/llama.cpp` must track:

`https://github.com/ggml-org/llama.cpp`

When updating vendor:

1.  Bump to commit that supports required GGUF.
2.  Run compatibility test.
3.  Fix binding only if necessary.

Never modify vendor code directly unless required for dependency mode.

------------------------------------------------------------------------

## ✅ Acceptance Test

This script must work:

``` python
from llama_cpp import Llama

llm = Llama(
    model_path="/path/to/Qwen3.5-0.8B-Q8_0.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    n_batch=512,
    use_mlock=True,
    use_mmap=False,
    chat_format=None,
    add_bos=True,
    add_eos=True,
    verbose=True,
)

stream = llm.create_chat_completion(
    messages=[{"role":"user","content":"hi"}],
    max_tokens=64,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    stream=True
)

for token in stream:
    print(token["choices"][0]["delta"].get("content",""), end="", flush=True)
```

Expected: - No architecture error - No CMake failure - No mtmd
configuration - Streaming works - Context persists until manually reset

------------------------------------------------------------------------

## 🚫 Explicitly Out of Scope

-   Embeddings
-   Multimodal
-   HTTP server mode
-   GPU backends
-   Vulkan
-   Metal
-   Windows wheels
-   Prebuilt distribution
-   API compatibility with external projects

This binding serves LYRN only.

------------------------------------------------------------------------

## 🔄 Update Strategy

When new Qwen releases appear:

1.  Update vendor llama.cpp
2.  Rebuild
3.  If architecture changed:
    -   Patch binding immediately
    -   Do NOT wait on upstream projects

This repo owns its compatibility.

------------------------------------------------------------------------

## 🧭 Philosophy

This is a controlled runtime component.

We prefer:

-   Stability over features
-   Determinism over convenience
-   Explicit control over automation
-   Small surface area over flexibility

This binding is infrastructure, not a feature platform.
