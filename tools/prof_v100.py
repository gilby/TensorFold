"""Kernel time breakdown of one verify window on the V100 (Qwen3.8-27B NVFP4)."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from tensorfold.families.qwen3_5.cuda.weights import load
from tensorfold.families.qwen3_5.cuda.forward import State, tree_forward
from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

w = load(sys.argv[1], "cuda")
st = State(w)
prefill_state(w, list(range(1000, 1300)), st)
for W in [int(a) for a in sys.argv[2:]] or [1, 16]:
    tok = torch.arange(5, 5 + W, device="cuda", dtype=torch.int32)
    par = [-1] + list(range(W - 1))
    for _ in range(3): tree_forward(w, tok, par, st)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(5): tree_forward(w, tok, par, st)
        torch.cuda.synchronize()
    ev = p.key_averages()
    rows = sorted([e for e in ev if e.device_type.name == "CUDA"], key=lambda e: -e.self_device_time_total)
    tot = sum(e.self_device_time_total for e in rows) / 5
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(5): tree_forward(w, tok, par, st)
    e.record(); torch.cuda.synchronize()
    print(f"== W={W}: wall {s.elapsed_time(e)/5:.1f} ms, gpu kernels {tot/1e3:.1f} ms")
    for r in rows[:25]:
        print(f"{r.self_device_time_total/5/1e3:8.2f} ms {r.count//5:5d}x  {r.key[:110]}")
