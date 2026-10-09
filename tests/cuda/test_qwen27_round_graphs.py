"""Several streams' tree-verify rounds replayed from RoundGraphs (sm_70 ranks' --parallel graphs) give every real
row the eager forward's bits: logits, records and drafter taps, bf16 and int8 caches, every GDN slot class."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.forward import GDNRecord, State, multi_tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.round_graphs import RoundGraphs  # noqa: E402
from tensorfold.families.qwen3_5.cuda.tree_graphs import slot_class  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, QLinear, Weights  # noqa: E402
from tensorfold.cuda.kernels import gdn as deltanet  # noqa: E402

CONFIG = Config(hidden=128, intermediate=256, layers=2, heads=2, kv_heads=2, head_dim=128, vocab=256, k_heads=2,
                v_heads=2, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32, rope_theta=10000000.0,
                eos=(0,))


def _inv_freq():
    half = CONFIG.rope_dims // 2
    return (CONFIG.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).float().cuda()


def _mlx_model() -> Weights:
    gen = torch.Generator(device="cuda").manual_seed(21)

    def q(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                              dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(q(768, 128), q(256, 128), q(2, 128), q(2, 128), q(128, 256),
              (torch.randn(768, 4, generator=gen, device="cuda") * 0.1).bfloat16(),
              torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), norm)
    attn = Attention(q(512, 128), q(256, 128), q(256, 128), q(128, 256), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, q(256, 128), q(256, 128), q(128, 256)),
              Layer(False, norm, norm, None, attn, q(256, 128), q(256, 128), q(128, 256))]
    w = Weights(CONFIG, q(256, 128), layers, norm, q(256, 128), _inv_freq(), tap_layers=(0, 1))
    return w


def _nvfp4_model() -> Weights:
    """NVFP4 MLP, FP8 GDN and attention projections, bf16 gates, embedding and head."""

    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear

    gen = torch.Generator().manual_seed(23)

    def fp4(n, k):
        packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).cuda()
        scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8).cuda()
        return Fp4Linear.from_checkpoint(packed, scale, 0.02)

    def fp8(n, k):
        raw = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
        raw[(raw & 0x7F) >= 0x60] = 0x30
        return Fp8Linear.from_checkpoint(raw.cuda().view(torch.float8_e4m3fn), 0.003)

    def bf16(n, k, scale=0.05):
        return Plain((torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16).cuda())

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(fp8(768, 128), fp8(256, 128), bf16(2, 128), bf16(2, 128), fp8(128, 256),
              (torch.randn(768, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda(),
              torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), norm)
    attn = Attention(fp8(512, 128), fp8(256, 128), fp8(256, 128), fp8(128, 256), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, fp4(256, 128), fp4(256, 128), fp4(128, 256)),
              Layer(False, norm, norm, None, attn, fp4(256, 128), fp4(256, 128), fp4(128, 256))]
    w = Weights(CONFIG, bf16(256, 128, 0.5), layers, norm, bf16(256, 128), _inv_freq(), quant="nvfp4",
                tap_layers=(0, 1))
    w.precision = "full"
    return w


def _volta_model() -> Weights:
    """``test_qwen27_nvfp4_tp_volta``'s two layers on the sm_70 linears (NVFP4 MLP, FP8 GDN/attention, bf16 gates and
    head), as the RadixArk 27B runs on Volta."""

    from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear

    gen = torch.Generator().manual_seed(11)

    def fp4(n, k):
        packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).cuda()
        scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8).cuda()
        return VoltaLinear.from_nvfp4(packed, scale.view(torch.float8_e4m3fn), 0.02)

    def fp8(n, k):
        raw = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
        raw[(raw & 0x7F) >= 0x60] = 0x30
        return VoltaLinear.from_fp8(raw.cuda().view(torch.float8_e4m3fn), 0.003)

    def bf16(n, k, scale=0.05):
        return VoltaLinear.from_bf16((torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16).cuda())

    n, m = 2, 256
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(fp8(3 * m, 128), fp8(m, 128), bf16(n, 128), bf16(n, 128), fp8(128, m),
              (torch.randn(3 * m, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda(),
              torch.zeros(n, device="cuda"), torch.zeros(n, device="cuda"), norm)
    attn = Attention(fp8(2 * m, 128), fp8(m, 128), fp8(m, 128), fp8(128, m), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, fp4(m, 128), fp4(m, 128), fp4(128, m)),
              Layer(False, norm, norm, None, attn, fp4(m, 128), fp4(m, 128), fp4(128, m))]
    embed = Plain((torch.randn(256, 128, generator=gen) * 0.5).to(torch.bfloat16).cuda())
    w = Weights(CONFIG, embed, layers, norm, bf16(256, 128), _inv_freq(), quant="nvfp4", tap_layers=(0, 1))
    w.precision = "full"
    return w


def _model() -> Weights:
    return _volta_model() if torch.cuda.get_device_capability()[0] == 7 else _nvfp4_model()


# chains, branching trees of each GDN slot class, a star, one row, and widths that pad (3 -> 4, 13 -> 16, 30 -> 32)
SHAPES = [[-1], [-1, 0, 1], list(range(-1, 11)), [-1, 0, 0, 1, 1, 2], [-1, 0, 1, 1, 0, 4, 5, 2],
          [-1] + [0] * 12, [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], list(range(-1, 29)),
          [-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7]]




def _tokens(n: int, seed: int) -> list[int]:
    return [3 + (seed * 31 + 17 * i) % 250 for i in range(n)]


@pytest.mark.parametrize("kv", ["bf16", "int8"])
def test_rounds_replay_equals_eager(kv):
    w = _model()
    w.kv_dtype = kv
    states = []
    for k, n in enumerate((9, 505, 40, 600)):
        st = State(w)
        prefill_state(w, _tokens(n, 30 + k), st)
        states.append(st)
    graphs = RoundGraphs(w, 4, 1024, tp=False, taps=True, full_logits=True)
    graphs.capture_all(State(w))
    shapes = [s for s in SHAPES if len(s) <= 16]
    seen = set()
    for rep in range(3):
        for S in (2, 3, 4):
            wins = []
            for j in range(S):
                parents = shapes[(rep * 5 + 3 * S + j) % len(shapes)]
                wins.append((_tokens(len(parents), 50 * rep + 7 * S + j), parents, states[j]))
            el, er, et, es = multi_tree_forward(w, wins, capture_taps=True)
            out = graphs.forward(wins)
            assert out is not None, (S, rep)
            seen.add(graphs.key(wins))
            gl, gr, gt, gs = out
            assert gs == es
            n = es[-1]
            assert torch.equal(el, gl[:n]), f"{S} streams, rep {rep}: logits differ"
            assert torch.equal(et, gt[:n]), "taps differ"
            for x, y in zip(er, gr):
                names = ("q", "k", "v", "g", "beta", "qkv") if isinstance(x, GDNRecord) else ("k", "v")
                for name in names:
                    assert torch.equal(getattr(x, name), getattr(y, name)[:n]), (S, rep, name)
    assert len({cls for _, cls in seen}) >= 2       # more than one GDN slot class replayed
    assert graphs.replays == 9


def test_a_round_that_fits_no_graph_runs_eagerly():
    w = _model()
    st = State(w)
    prefill_state(w, _tokens(20, 3), st)
    graphs = RoundGraphs(w, 2, 64, tp=False, taps=True, full_logits=True, widths=(4, 8))
    lone = [(_tokens(3, 1), [-1, 0, 1], st)]
    wide = [(_tokens(6, 1), list(range(-1, 5)), st), (_tokens(6, 2), list(range(-1, 5)), st)]
    far = [(_tokens(3, 1), [-1, 0, 1], st), (_tokens(3, 1), [-1, 0, 1], st)]
    assert graphs.key(lone) is None                  # one stream: the lone-stream graphs' round
    assert graphs.key(wide) is None                  # 12 rows past the widest graph
    assert graphs.key(far) == (8, slot_class(0))
    graphs.context = 20                              # keys past the graphs' context
    assert graphs.key(far) is None
    assert slot_class(deltanet.schedule([-1, 0, 0])[1]) in graphs.classes
