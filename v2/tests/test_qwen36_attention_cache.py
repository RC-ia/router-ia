from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router_ia.qwen36_attention_cache import AttentionState, _append_full_kv


def _token() -> tuple[torch.Tensor, torch.Tensor]:
    # (batch, kv_heads, sequence, head_dim), matching the cache layout.
    return torch.zeros((1, 1, 1, 2)), torch.zeros((1, 1, 1, 2))


def test_full_kv_growth_stops_before_exceeding_context_limit() -> None:
    state = AttentionState(max_context_tokens=3)

    for absolute_position in range(3):
        state.tokens_seen = absolute_position
        key, value = _token()
        _append_full_kv(state, 0, key, value)
        assert state.full_keys[0].shape[-2] <= 3
        assert state.full_values[0].shape[-2] <= 3

    state.tokens_seen = 3
    key, value = _token()
    with pytest.raises(RuntimeError, match=r"max_context_tokens=3"):
        _append_full_kv(state, 0, key, value)

    stats = state.snapshot()
    assert state.full_keys[0].shape[-2] == 3
    assert stats["absolute_position"] == 3
    assert stats["full_tokens_resident"] == 3
    assert stats["kv_start_position"] == 0


def test_full_kv_growth_stops_before_exceeding_byte_budget() -> None:
    key, value = _token()
    token_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
    state = AttentionState(max_full_kv_bytes=token_bytes * 2)

    for absolute_position in range(2):
        state.tokens_seen = absolute_position
        _append_full_kv(state, 0, key, value)
        assert state.snapshot()["full_bytes"] <= token_bytes * 2

    state.tokens_seen = 2
    with pytest.raises(RuntimeError, match=r"max_full_kv_bytes"):
        _append_full_kv(state, 0, key, value)

    assert state.full_keys[0].shape[-2] == 2
    assert state.snapshot()["full_bytes"] == token_bytes * 2
