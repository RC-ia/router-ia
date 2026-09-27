from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from router_ia import qwen36_chat_batch as chat


class _Tokenizer:
    eos_token_id = None

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return f"<{token_ids[0]}>"


class _TemplateTokenizer(_Tokenizer):
    chat_template = "{{ messages }}{% if add_generation_prompt %}<assistant>{% endif %}"

    def __init__(self) -> None:
        self.apply_chat_template = Mock(return_value=[101, 102, 103])


def test_prompt_token_ids_uses_chat_template_with_generation_marker() -> None:
    tokenizer = _TemplateTokenizer()

    prompt_ids = chat.prompt_token_ids(tokenizer, "Olá")

    assert prompt_ids == [101, 102, 103]
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "Olá"}],
        tokenize=True,
        add_generation_prompt=True,
        tokenizer_kwargs={"add_special_tokens": False},
    )


def test_prompt_token_ids_falls_back_to_simple_prompt_without_template() -> None:
    tokenizer = _Tokenizer()

    prompt_ids = chat.prompt_token_ids(tokenizer, "prompt simples")

    assert prompt_ids == [1, 2]


def test_prompt_token_ids_raw_prompt_bypasses_available_template() -> None:
    tokenizer = _TemplateTokenizer()

    prompt_ids = chat.prompt_token_ids(tokenizer, "prompt bruto", raw_prompt=True)

    assert prompt_ids == [1, 2]
    tokenizer.apply_chat_template.assert_not_called()


@pytest.fixture
def generation_mocks(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(reset=Mock())
    monkeypatch.setattr(chat.attention_cache, "state_for", Mock(return_value=state))
    monkeypatch.setattr(chat.attention_cache, "activate", Mock())
    monkeypatch.setattr(chat.attention_cache, "deactivate", Mock())
    monkeypatch.setattr(chat.attention_cache, "stats", Mock(return_value={"tokens_seen": 2, "full_tokens": 0, "bytes": 0}))
    monkeypatch.setattr(chat, "cache_stats", Mock(return_value={}))
    monkeypatch.setattr(chat, "print_cache", Mock())
    monkeypatch.setattr(chat, "print_attention", Mock())
    monkeypatch.setattr(chat, "run_forward_token", Mock(return_value=(object(), 0.0, 0.0)))
    return state


@pytest.mark.parametrize(
    ("max_new_tokens", "expected_tokens", "expected_sample_calls"),
    [
        (0, [], 0),
        (1, [10], 1),
        (3, [10, 11, 12], 3),
    ],
)
def test_generate_response_respects_max_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
    generation_mocks,
    capsys: pytest.CaptureFixture[str],
    max_new_tokens: int,
    expected_tokens: list[int],
    expected_sample_calls: int,
) -> None:
    sampled_ids = iter([10, 11, 12])
    sample_next = Mock(side_effect=lambda *_: next(sampled_ids))
    generated_step = Mock(side_effect=lambda token_id, *_: (sample_next(object()), 0.0, 0.0))
    monkeypatch.setattr(chat, "sample_next", sample_next)
    monkeypatch.setattr(chat, "run_generated_token", generated_step)

    generated = chat.generate_response(
        Path("/model"),
        "prompt",
        _Tokenizer(),
        None,
        None,
        "final_norm",
        "lm_head",
        "cpu",
        max_new_tokens,
        20,
        0.0,
    )

    assert generated == expected_tokens
    assert sample_next.call_count == expected_sample_calls
    assert generated_step.call_count == max(max_new_tokens - 1, 0)
    assert capsys.readouterr().out.count("<") == len(expected_tokens)


def test_non_negative_int_rejects_negative_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        chat.non_negative_int("-1")


def test_router_state_reset_when_switching_to_checkpoint_without_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from router_ia import qwen36_chat_batch_fused as fused

    first_root = tmp_path / "first-model"
    second_root = tmp_path / "second-model"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "model.safetensors.index.json").write_text('{"model": "first"}')
    (second_root / "model.safetensors.index.json").write_text('{"model": "second"}')
    monkeypatch.chdir(tmp_path)
    predictor = fused.ExpertCrossLayerPredictor()
    monkeypatch.setattr(fused, "_ROUTING_PREDICTOR", predictor)
    monkeypatch.setattr(fused, "_ROUTER_STATE_ROOT", None)
    monkeypatch.setattr(fused, "_ROUTER_STATE_PATH", None)
    monkeypatch.setattr(fused, "_ROUTER_STATE_LOADED", False)

    predictor.observe(2, [22], {0: (11,)})
    monkeypatch.setattr(fused, "_ROUTER_STATE_ROOT", first_root.resolve())
    predictor.save(fused._router_state_path(first_root))
    monkeypatch.setattr(fused, "_ROUTER_STATE_ROOT", None)

    fused._ensure_router_state(first_root)
    assert predictor.predict(2, {0: (11,)}) == [22]

    assert not fused._router_state_path(second_root).exists()
    fused._ensure_router_state(second_root)

    assert not fused._ROUTER_STATE_LOADED
    assert predictor.predict(2, {0: (11,)}) == []

@pytest.mark.parametrize("value", ["cpu", "cuda", "cuda:0"])
def test_device_parser_accepts_pytorch_device_values(value: str) -> None:
    args = chat.build_parser().parse_args(["/model", "--device", value])

    assert str(args.device) == value


def test_device_parser_rejects_invalid_pytorch_device() -> None:
    with pytest.raises(SystemExit):
        chat.build_parser().parse_args(["/model", "--device", "not-a-device"])


def test_indexed_cuda_device_uses_cuda_vram_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from router_ia import qwen36_cached_loop as cached

    properties = SimpleNamespace(total_memory=8 * 1024**3)
    get_properties = Mock(return_value=properties)
    set_fraction = Mock()
    monkeypatch.setattr(cached, "VRAM_GB", 1.0)
    monkeypatch.setattr(cached.torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(cached.torch.cuda, "get_device_properties", get_properties)
    monkeypatch.setattr(cached.torch.cuda, "set_per_process_memory_fraction", set_fraction)

    cached._configure_vram_limit(chat.torch.device("cuda:3"))

    get_properties.assert_called_once_with(3)
    set_fraction.assert_called_once_with(pytest.approx(1 / 8), 3)
