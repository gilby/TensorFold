import sys, torch
from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear
n, k, m, fmt = (int(a) for a in sys.argv[1:5])
if fmt == 0:
    lin = VoltaLinear.from_nvfp4(torch.randint(0, 255, (n, k // 2), device="cuda", dtype=torch.int32).to(torch.uint8),
                                 torch.full((n, k // 16), 0x30, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn), 0.01)
else:
    lin = VoltaLinear.from_fp8(torch.full((n, k), 0x30, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn), 0.01)
x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
for _ in range(3): lin(x)
torch.cuda.synchronize()
