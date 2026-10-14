"""A Qwen3.6 MoE MTP layer stored in mixed lane formats (oQ's per-module widths) loads each projection at its own."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] == 7:
    pytest.skip("sm_70 serves NVFP4 MoE checkpoints only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import dequantize  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import matmul, untile  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Config  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.weights import load_mtp  # noqa: E402

D, E, WIDTH, TOP, HD = 256, 8, 64, 2, 128
FORMATS = {"fc": 6, "layers.0.self_attn.q_proj": 8, "layers.0.self_attn.k_proj": 5, "layers.0.self_attn.v_proj": 5,
           "layers.0.self_attn.o_proj": 8, "layers.0.mlp.gate": 8, "layers.0.mlp.shared_expert_gate": 8}


def _affine(gen, shape, bits):
    *lead, n, k = shape
    codes = torch.randint(0, 1 << bits, (*lead, n, k), generator=gen, dtype=torch.int32)
    words = qmm.words_of(codes.reshape(-1, k), bits).reshape(*lead, n, k * bits // 32)
    s = (torch.rand((*lead, n, k // 64), generator=gen) * 0.01 + 0.001).bfloat16()
    b = (torch.rand((*lead, n, k // 64), generator=gen) * 0.01 - 0.005).bfloat16()
    return words, s, b


@pytest.mark.parametrize("bf16_fc", [False, True])
def test_mtp_projections_take_their_declared_formats(tmp_path, bf16_fc):
    gen = torch.Generator().manual_seed(3)
    tensors, quant = {}, {"group_size": 64, "bits": 4, "mode": "affine"}
    shapes = {"fc": (D, 2 * D), "layers.0.self_attn.q_proj": (2 * 2 * HD, D), "layers.0.self_attn.k_proj": (HD, D),
              "layers.0.self_attn.v_proj": (HD, D), "layers.0.self_attn.o_proj": (D, 2 * HD),
              "layers.0.mlp.gate": (E, D), "layers.0.mlp.shared_expert_gate": (1, D),
              "layers.0.mlp.switch_mlp.gate_proj": (E, WIDTH, D), "layers.0.mlp.switch_mlp.up_proj": (E, WIDTH, D),
              "layers.0.mlp.switch_mlp.down_proj": (E, D, WIDTH), "layers.0.mlp.shared_expert.gate_proj": (WIDTH, D),
              "layers.0.mlp.shared_expert.up_proj": (WIDTH, D), "layers.0.mlp.shared_expert.down_proj": (D, WIDTH)}
    for name, shape in shapes.items():
        bits = FORMATS.get(name, 4)
        w, s, b = _affine(gen, shape, bits)
        key = "language_model.mtp." + name
        tensors.update({key + ".weight": w, key + ".scales": s, key + ".biases": b})
        if bits != 4:
            quant[key] = {"bits": bits, "group_size": 64, "mode": "affine"}
    if bf16_fc:                                      # oQ keeps the fc bf16 beside its quantized layer
        for suffix in (".scales", ".biases"):
            del tensors["language_model.mtp.fc" + suffix]
        tensors["language_model.mtp.fc.weight"] = (torch.randn((D, 2 * D), generator=gen) * 0.02).bfloat16()
        del quant["language_model.mtp.fc"]
    for norm in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm", "layers.0.input_layernorm",
                 "layers.0.post_attention_layernorm", "layers.0.self_attn.q_norm", "layers.0.self_attn.k_norm"):
        width = HD if norm.endswith(("q_norm", "k_norm")) else D
        tensors[f"language_model.mtp.{norm}.weight"] = torch.ones(width, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model-mtp.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-mtp.safetensors" for k in tensors}}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "quantization": quant}))

    class W:
        quant = "mlx"
        config = Config(hidden=D, intermediate=0, layers=2, heads=2, kv_heads=1, head_dim=HD, vocab=512, k_heads=1,
                        v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                        rope_theta=1e7, eos=(0,), experts=E, top_k=TOP, moe_width=WIDTH)

    m = load_mtp(tmp_path, W())
    x = torch.randn((5, D), device="cuda").bfloat16()
    attn = {"q_proj": m.attn.q, "k_proj": m.attn.k, "v_proj": m.attn.v}
    for leaf, q in attn.items():
        name = "layers.0.self_attn." + leaf
        assert (q.bits, q.gs, q.layout) == (FORMATS[name], 64, "tiled"), name
        key = "language_model.mtp." + name
        ref = dequantize(tensors[key + ".weight"].cuda(), tensors[key + ".scales"].cuda(),
                         tensors[key + ".biases"].cuda(), FORMATS[name], 64)
        assert torch.equal(dequantize(*[t for t in (untile(q).weight, untile(q).scales, untile(q).biases)],
                                      q.bits, 64), ref), name
    if bf16_fc:
        fc = tensors["language_model.mtp.fc.weight"].cuda()
        for half, q in ((fc[:, :D], m.fc_e), (fc[:, D:], m.fc_h)):
            assert q.layout == "b16" and torch.equal(q.weight, half)
        fc = fc.float()
    else:
        fc = dequantize(*(tensors["language_model.mtp.fc" + s].cuda() for s in (".weight", ".scales", ".biases")), 6, 64)
    for half, q in ((fc[:, :D], m.fc_e), (fc[:, D:], m.fc_h)):
        if not bf16_fc:
            assert (q.bits, q.n, q.k) == (6, D, D)
            assert torch.equal(dequantize(untile(q).weight, untile(q).scales, untile(q).biases, 6, 64), half)
        y = (matmul(x, q) if not bf16_fc else q(x)).float()
        assert (y - x.float() @ half.T).abs().max().item() <= (x.float() @ half.T).abs().max().item() * 2 ** -7
    assert m.moe.router.shape == (E + 1, D) and m.moe.experts.count == E + 1
