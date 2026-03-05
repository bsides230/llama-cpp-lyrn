import numpy as np

from llama_cpp.llama import LlamaState
from llama_cpp.llama_cache import LlamaRAMCache


def _state_for_tokens(tokens):
    input_ids = np.array(tokens + [0, 0], dtype=np.intc)
    scores = np.zeros((len(input_ids), 1), dtype=np.single)
    return LlamaState(
        input_ids=input_ids,
        scores=scores,
        n_tokens=len(tokens),
        llama_state=b"state",
        llama_state_size=1,
        seed=0,
    )


def test_ram_cache_uses_full_prefix_entries_only():
    cache = LlamaRAMCache()

    # Divergent key: [1,2,3,4] is not a prefix of [1,2,7]
    cache[(1, 2, 3, 4)] = _state_for_tokens([1, 2, 3, 4])

    # Exact prefix key should be selected instead.
    expected = _state_for_tokens([1, 2])
    cache[(1, 2)] = expected

    result = cache[(1, 2, 7)]

    assert result.n_tokens == expected.n_tokens
    assert result.input_ids[: expected.n_tokens].tolist() == [1, 2]


def test_ram_cache_returns_key_error_when_only_partial_overlap_exists():
    cache = LlamaRAMCache()
    cache[(1, 2, 3)] = _state_for_tokens([1, 2, 3])

    # Overlap exists (token 1), but no full-prefix key exists for the prompt.
    # Cache should miss instead of returning a divergent state.
    try:
        _ = cache[(1, 9, 9)]
        assert False, "expected cache miss"
    except KeyError:
        pass
