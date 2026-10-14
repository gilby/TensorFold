# TensorFold-friendly oQ8e for Gemma-4-12B-it (gemma4_unified), 2026-10-02.
# oQ level 8, enhanced (imatrix), every quantized tensor in groups of 64: the gemma4_unified family's decode kernels
# read any affine width 2-8 at g64 (lane matmul / simd_qmm_bits), so widths stay oQ's own; only the group is pinned.
# Run with oMLX's python: /opt/homebrew/Cellar/omlx/HEAD-3f2d07e/libexec/bin/python oq_tf_friendly_gemma12.py
import os, time, json
import omlx.oq as oq

_orig_pred = oq.universal_quant_predicate


def _pred(path, module, config, oq_level=8):
    r = _orig_pred(path, module, config, oq_level)
    if isinstance(r, dict):
        return {**r, "group_size": 64, "mode": "affine"}
    return r


oq.universal_quant_predicate = _pred
SRC = os.path.expanduser("~/models/google/gemma-4-12B-it")
OUT = os.path.expanduser("~/models/gemma-4-12B-it-oQ8e")
t0 = time.time(); last = [0.0]


def cb(phase, pct, detail="", meta=None):
    if pct - last[0] >= 2 or "sav" in phase:
        last[0] = pct; print(f"{time.time()-t0:7.0f}s {pct:5.1f}% {phase} {detail}", flush=True)


oq.quantize_oq_streaming(SRC, OUT, 8, 64, cb, False, None, None, "", "bfloat16", False, True,
                         enhanced=True, imatrix_num_samples=128, imatrix_seq_length=512)
q = json.load(open(OUT + "/config.json")); q = q.get("quantization") or q.get("quantization_config")
base = {k: v for k, v in q.items() if not isinstance(v, dict)}
over = {}
for k, v in q.items():
    if not isinstance(v, dict): continue
    key = (v["bits"], v.get("group_size"), v.get("mode", "affine"))
    over[key] = over.get(key, 0) + 1
print("DONE", round(time.time() - t0), "s  base", base, "overrides", over, flush=True)
bad = [k for k in over if k[1] != 64 or k[2] != "affine"] + ([] if base.get("group_size") == 64 else ["BASE"])
print("CHECK", "OK: all affine g64" if not bad else f"FAIL {bad}", flush=True)
