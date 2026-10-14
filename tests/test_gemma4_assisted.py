"""Gemma 4 (26B MoE and 31B dense layouts) drafting from an MTP assistant: drafted == serial, concurrent == alone."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.gemma4_text")
if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from gemma4_tiny import tiny_text, tokens  # noqa: E402
from gemma4_unified_tiny import ASSISTANT, tiny_assistant  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.gemma4.assisted import Gemma4Assisted  # noqa: E402
from tensorfold.families.gemma4.model import Gemma4, is_assistant  # noqa: E402
from tensorfold.families.gemma4_unified.assistant import Assistant  # noqa: E402

BACKENDS = ["rows", "auto"]


@pytest.fixture(scope="module", params=("dense", "dense-oq", "moe"))
def text(request):
    return tiny_text(kind=request.param)


def plain(text, backend: str) -> Gemma4:
    model = Gemma4(text, backend=backend, check=False)
    model.exact_width = 16
    return model


def assisted(text, backend: str, drafter) -> Gemma4Assisted:
    model = Gemma4Assisted(text, backend=backend, check=False, drafter=drafter)
    model.exact_width = 16
    return model


def reply(model, prompt, n, *, sampling=None, drafts=False):
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling, drafts=drafts)
    engine.add_stream(stream)
    engine.run()
    return list(stream.emitted), engine


class Known(Assistant):
    """The real assistant forward, then a known reply's tokens (wrong from ``good`` on), so some drafts are kept."""

    def __init__(self, inner: Assistant, sequence: list[int], good=(3, 0, 5, 1, 9)) -> None:
        self.__dict__.update(inner.__dict__)
        self.sequence, self.good, self.calls = list(sequence), good, 0

    def draft(self, layers, hidden, token, row_position, position, sampling, count):
        mx.eval(super().draft(layers, hidden, token, row_position, position, sampling, count))
        good = self.good[self.calls % len(self.good)]
        self.calls += 1
        out = [self.sequence[position + s] if position + s < len(self.sequence) else 0 for s in range(count)]
        return mx.array([t if s < good else (t + 1 + s) % 256 for s, t in enumerate(out)], dtype=mx.uint32)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampled", [False, True])
def test_assistant_drafted_replies_equal_serial_ones(text, backend, sampled):
    prompt = tokens(150, seed=21)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = reply(plain(text, backend), prompt, 40, sampling=sampling)
    model = assisted(text, backend, Known(tiny_assistant(), prompt + serial))
    assert model.mtp is not None and model.drafts == Assistant.default_drafts
    drafted, engine = reply(model, prompt, 40, sampling=sampling, drafts=True)
    assert drafted == serial
    assert engine.drafted > 0 and 0 < engine.accepted < engine.drafted
    # the real (random) assistant: whatever it drafts, the reply is the serial one
    assert reply(assisted(text, backend, tiny_assistant()), prompt, 40, sampling=sampling, drafts=True)[0] == serial


@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_assisted_streams_emit_what_they_emit_alone(text, backend):
    prompts = [tokens(n, seed=30 + n) for n in (25, 150, 60, 9)]
    samplings = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95), None,
                 Sampling(seed=8, temperature=0.8, top_k=40, top_p=0.9)]
    alone = [reply(plain(text, backend), p, 20, sampling=s)[0] for p, s in zip(prompts, samplings)]
    model = assisted(text, backend, tiny_assistant())
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    streams = []
    for i, (p, s) in enumerate(zip(prompts, samplings)):
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=20, sampling=s, drafts=i % 2 == 0)
        engine.add_stream(stream)
        streams.append(stream)
    engine.run()
    assert [list(s.emitted) for s in streams] == alone
    assert any(r.streams > 1 for r in engine.round_stats)


def test_a_drafter_directory_is_told_apart_by_its_model_type(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text(json.dumps(dict(ASSISTANT, model_type="gemma4_assistant")))
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "config.json").write_text(json.dumps({"model_type": "qwen3", "block_size": 16}))
    assert is_assistant(tmp_path / "a") and not is_assistant(tmp_path / "d") and not is_assistant(tmp_path / "x")
