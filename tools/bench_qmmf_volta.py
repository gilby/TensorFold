"""GB/s of the Volta decode matmuls (NVFP4 / FP8 / 16-bit, and the 4-bit affine one) at verify row counts."""
import sys, torch
from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear, FP4, FP8, F16
from tensorfold.cuda.kernels import qmm_volta

SH = [("gate|up", 17408, 5120), ("down", 5120, 17408), ("qkv", 10240, 5120), ("z", 6144, 5120), ("out", 5120, 6144),
      ("head", 248320, 5120)]
def t(fn, it=50):
    for _ in range(3): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3
rows = [int(r) for r in (sys.argv[1:] or ["1", "8", "16", "32"])]
for name, n, k in SH:
    for fmt, label in ((FP4, "nvfp4"), (FP8, "fp8"), (F16, "f16"), (-1, "affine4")):
        if name == "head" and fmt not in (F16, FP4): continue
        if fmt == FP4:
            lin = VoltaLinear.from_nvfp4(torch.randint(0, 255, (n, k // 2), device="cuda", dtype=torch.int32).to(torch.uint8),
                                         torch.full((n, k // 16), 0x30, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn), 0.01)
        elif fmt == FP8:
            lin = VoltaLinear.from_fp8(torch.full((n, k), 0x30, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn), 0.01)
        elif fmt == F16:
            lin = VoltaLinear.from_bf16((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16))
        else:
            words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), device="cuda", dtype=torch.int64).to(torch.int32)
            s = torch.full((n, k // 64), 0.01, device="cuda", dtype=torch.bfloat16)
            tl = qmm_volta.Tiled(words, s, s)
            lin = lambda x, tl=tl: qmm_volta.tiled_matmul(x, tl)
        nbytes = lin.nbytes() if fmt >= 0 else (tl.weight.numel() * 4 + tl.scales.numel() * 4)
        out = []
        for m in rows:
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            us = t(lambda: lin(x))
            out.append(f"M={m}: {us:7.1f}us {nbytes / us / 1e3:6.0f}GB/s")
        print(f"{name:8s} {n:6d}x{k:5d} {label:8s}", " | ".join(out), flush=True)
        del lin
        torch.cuda.empty_cache()
