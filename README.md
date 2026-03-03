# llama-cpp-lyrn --- Task Prompt for Jules

This is a working implementation task.

We are building `llama-cpp-lyrn`: a minimal, deterministic Python
binding around llama.cpp for LYRN.

Keep it simple. Keep it tight. No feature creep.

------------------------------------------------------------------------

## Repo Structure (Current)

repo-root/ │ ├── .git/ ├── llama-cpp-python source/ (backend source
only) ├── model_runner.py (defines actual runtime usage) └── README.md

Important:

• The repo root is the build root. • `llama-cpp-python source/` is
backend source only. • Do NOT build inside that folder directly. • All
packaging, CMake control, and wheel logic must be owned from repo root.

------------------------------------------------------------------------

## Scope Definition (Explicit)

The binding surface must be derived directly from:

    model_runner.py

Do NOT guess what LYRN uses. Do NOT preserve upstream features "just in
case."

Only keep functionality that is:

• Imported in model_runner.py • Instantiated in model_runner.py • Called
in model_runner.py • Required to make model_runner.py function correctly

Everything else is removed or disabled.

If model_runner.py does not use it, it does not belong in this binding.

------------------------------------------------------------------------

## What model_runner.py Requires

The binding must support:

1.  `from llama_cpp import Llama`
2.  Constructor with these parameters:
    -   model_path
    -   n_ctx
    -   n_threads
    -   n_gpu_layers
    -   n_batch
    -   use_mlock
    -   use_mmap
    -   chat_format
    -   add_bos
    -   add_eos
    -   verbose
3.  `create_chat_completion(..., stream=True)`
4.  Streaming output in this exact shape:

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

No alternate streaming schema allowed.

model_runner.py captures stderr for metrics parsing. Do NOT suppress
stderr output from llama.cpp. Do NOT redirect logs internally unless
explicitly required.

------------------------------------------------------------------------

## Error Handling & Debugging Requirements

This layer must remain debuggable.

Rules:

• Do not swallow exceptions silently. • If model loading fails, raise
clear Python exceptions. • If generation fails, propagate meaningful
error messages. • Preserve llama.cpp stderr output for external capture.
• Do not wrap errors in generic catch-all handlers. • Avoid hidden
fallback behavior that masks backend failures. • If GPU backend fails to
initialize, log the reason clearly. • If architecture mismatch occurs
(e.g., unknown GGUF architecture), raise explicit error.

When debugging:

• Ensure CMake configuration errors surface clearly. • Ensure backend
selection (CPU vs GPU) is visible in logs. • Ensure build flags can be
verified from verbose output. • Ensure wheel backend variant can be
determined at runtime (e.g., via logging or attribute).

Determinism includes deterministic failure modes.

Errors must be clear and actionable.

------------------------------------------------------------------------

## Build Rules

Default build:

• CPU-only • No server • No tools • No tests • No examples • No
multimodal • No embeddings

GPU support:

• Provided via prebuilt wheels • Users must NOT rebuild manually •
Backend selected via wheel at install time • Runtime GPU activation when
`n_gpu_layers > 0`

------------------------------------------------------------------------

## Qwen 3.5 Support

Must load GGUF models where:

    general.architecture = qwen35

No "unknown model architecture" errors allowed.

------------------------------------------------------------------------

## Context Behavior (As Used by model_runner.py)

• Model stays loaded • Context persists across calls • No implicit
resets • No hidden lifecycle changes • Binding must not interfere with
snapshot/delta logic

model_runner.py controls lifecycle.

The binding only provides inference.

------------------------------------------------------------------------

That is the contract.

Implement only what model_runner.py requires. Remove everything else.
Keep it deterministic.
