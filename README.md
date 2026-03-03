
# llama-cpp-lyrn — Implementation Directive (API Alignment Update)

This is a working implementation task for building `llama-cpp-lyrn`.

This document is intentionally explicit so no clarification questions are required.

We are building a minimal, deterministic Python binding around CURRENT upstream llama.cpp
for use exclusively by `model_runner.py` inside LYRN.

Keep it minimal. Keep it controlled. No feature creep.

---

## 1️⃣ Build Ownership & Layout

• Repo root is the build root.
• All packaging, CMake configuration, and wheel logic live at repo root.
• Create new `pyproject.toml` and `CMakeLists.txt` at repo root.
• Backend C/C++ source is referenced from the root build.

The folder:

    llama-cpp-python source/

is reference-only.

Do not treat it as canonical.
Do not preserve it blindly.
Do not inherit legacy binding logic.

---

## 2️⃣ Upstream Alignment (Non-Negotiable)

We target the CURRENT upstream llama.cpp API.

We do NOT:

• Maintain compatibility with old symbols
• Recreate deprecated functions
• Patch missing symbols like `llama_get_kv_self`
• Carry forward obsolete ctypes mappings

If upstream removed a symbol, it must NOT exist in our binding.

Bindings must match the current header definitions exactly.

This is a forward-only binding.

---

## 3️⃣ Binding Strategy (Reductionist Rebuild)

We are NOT fixing 200+ legacy ctypes bindings.

We are:

• Rebuilding a minimal binding layer
• Defining only the ctypes mappings strictly required
• Mapping only the functions used by `Llama.__init__`
• Mapping only the functions required by `create_chat_completion`
• Removing all other ctypes references

Do NOT try to “repair” the old binding set.

Strip aggressively and rebuild against the current llama.cpp headers.

---

## 4️⃣ Scope Definition (Derived from model_runner.py)

The binding surface is derived strictly from:

    model_runner.py

Keep only what is required for:

• Constructor parameters used there
• `create_chat_completion(..., stream=True)`
• Streaming output format used there
• stderr log behavior required for metric parsing

Everything else is removed.

---

## 5️⃣ Required Public API (Stable Surface)

from llama_cpp import Llama

Constructor:

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

Required method:

create_chat_completion(..., stream: bool)

Streaming MUST yield:

{
    "choices": [
        {
            "delta": {
                "content": "..."
            }
        }
    ]
}

No alternate streaming schema.

stderr output must remain visible.

---

## 6️⃣ CMake Configuration Rules

Root CMake must:

• Compile llama.cpp as shared library
• Disable server
• Disable tools
• Disable examples
• Disable tests
• Disable multimodal
• Build only inference components

This build is system-specific and not intended for general reuse.

---

## 7️⃣ Qwen 3.5 Requirement

Must load GGUF models where:

    general.architecture = qwen35

Use current upstream llama.cpp that supports this.

Do not attempt to reintroduce legacy compatibility layers.

---

## 8️⃣ Context Behavior

• Model remains loaded
• Context persists
• No implicit resets
• Binding must not override lifecycle logic

`model_runner.py` owns lifecycle.

---

## 9️⃣ Error Handling & Debugging

• Do not swallow exceptions.
• Surface symbol resolution errors clearly.
• Surface backend mismatches clearly.
• Preserve stderr output.
• Failure modes must be deterministic.

---

This is a forward-only minimal binding.

We rebuild against current llama.cpp.
We do not preserve legacy API compatibility.
We strip everything not required.

Implement incrementally and test continuously.
