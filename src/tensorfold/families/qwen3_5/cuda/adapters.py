"""Runtime LoRA adapters on the 27B's prompt path (Cloudflare Clef): each adapted projection adds x A^T B^T while a
decision request prefills, so chat, which never runs with an adapter on, keeps its bits.

An adapter directory holds ``adapter.safetensors`` (``layers.{L}.{proj}.lora_A`` [r, in] and ``.lora_B`` [out, r],
bf16, the scale folded into B) and ``manifest.json``. Each rank keeps the factors of its own shard, cut like the base
projection (``distributed.split_layer``): a column-parallel projection's B by the same output rows (A whole); a
row-parallel one's A by the same input columns (B whole), so the rank-ordered sum of the partials carries x A^T B^T.

Both products keep a row's bits independent of how many rows a chunk or a batch holds (prompt chunking and batching
keep the same decisions): on sm_70 the pinned per-shape cuBLASLt prompt GEMM of a 16-bit ``VoltaLinear``; elsewhere
fixed-shape GEMMs a row block at a time.
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path

import torch

ACTIVE = None             # the adapter prompt matmuls apply now (set only around a decision prefill), else None
BLOCK = 256               # rows per LoRA GEMM (fixed shape: row bits independent of the chunk or batch)
_LOCK = threading.Lock()

# manifest projection -> (owner attribute path, parallel kind, split arguments from the config)
_PROJ = {
    "linear_attn.in_proj_qkv": ("gdn.qkv", "col", lambda c: {"segments": (c.k_heads * c.dk, c.k_heads * c.dk,
                                                                          c.v_heads * c.dv)}),
    "linear_attn.in_proj_z": ("gdn.z", "col", lambda c: {"block": c.dv}),
    "linear_attn.in_proj_b": ("gdn.b", "col", lambda c: {}),
    "linear_attn.in_proj_a": ("gdn.a", "col", lambda c: {}),
    "linear_attn.out_proj": ("gdn.out", "row", lambda c: {}),
    "self_attn.q_proj": ("attn.q", "col", lambda c: {"block": 2 * c.head_dim}),
    "self_attn.k_proj": ("attn.k", "col", lambda c: {"block": c.head_dim}),
    "self_attn.v_proj": ("attn.v", "col", lambda c: {"block": c.head_dim}),
    "self_attn.o_proj": ("attn.o", "row", lambda c: {}),
    "mlp.gate_proj": ("gate", "col", lambda c: {}),
    "mlp.up_proj": ("up", "col", lambda c: {}),
    "mlp.down_proj": ("down", "row", lambda c: {}),
}


def _blocked(x: torch.Tensor, fn, n_out: int) -> torch.Tensor:
    """``fn`` over fixed ``BLOCK``-row tiles of ``x`` (the last zero-padded): fp32 (M, n_out)."""

    m = x.shape[0]
    out = torch.empty((m, n_out), dtype=torch.float32, device=x.device)
    tile = torch.empty((BLOCK, x.shape[1]), dtype=x.dtype, device=x.device)   # one buffer: the GEMM never sees
    for a in range(0, m, BLOCK):                                              # another address or alignment
        b = min(m, a + BLOCK)
        tile[:b - a] = x[a:b]
        if b - a < BLOCK:
            tile[b - a:] = 0
        out[a:b] = fn(tile)[:b - a]
    return out


class _Dense:
    """One factor as a bf16 matrix: fp32 products of fixed-shape tiles (sm_80 and newer, and CPU tests)."""

    def __init__(self, m: torch.Tensor):
        self.m = m.contiguous()
        self.n = m.shape[0]

    def nbytes(self) -> int:
        return self.m.numel() * self.m.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return _blocked(x.to(torch.bfloat16), lambda t: torch.matmul(t, self.m.t()).float(), self.n)
        return _blocked(x.float(), lambda t: t @ self.m.float().t(), self.n)


class _Volta:
    """One factor on sm_70: the 16-bit projection the bf16 checkpoint layers use (``VoltaLinear.from_bf16``, a power of
    two per output), its pinned cuBLASLt prompt GEMM, chunk invariant; fp32 out."""

    def __init__(self, m: torch.Tensor):
        from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear

        self.lin = VoltaLinear.from_bf16(m.contiguous())
        self.n = m.shape[0]

    def nbytes(self) -> int:
        return self.lin.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # the pinned per-shape GEMM is row-count invariant already (qmm_volta.prompt_matmul): no row blocks needed
        return self.lin.prefill(x.to(torch.bfloat16), f32=True)


class Lora:
    """A projection shard's two factors: (M, in) rows -> fp32 (M, out) = (x A^T) B^T, the middle rounded to bf16."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor, volta: bool):
        kind = _Volta if volta else _Dense
        pad = -a.shape[0] % 64                    # a rank of whole 64-input groups (zero rows add nothing)
        if pad:
            a = torch.cat([a, a.new_zeros((pad, a.shape[1]))])
            b = torch.cat([b, b.new_zeros((b.shape[0], pad))], dim=1)
        self.a, self.b = kind(a), kind(b)
        self.rank = a.shape[0]

    def nbytes(self) -> int:
        return self.a.nbytes() + self.b.nbytes()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x).to(torch.bfloat16))


class Adapter:
    """An adapter's factors for this rank's projections, keyed by the projection objects themselves."""

    def __init__(self, name: str, mods: dict, owners: list, manifest: dict):
        self.name, self.mods, self._owners, self.manifest = name, mods, owners, manifest

    def nbytes(self) -> int:
        return sum(m.nbytes() for m in self.mods.values())

    def has(self, w) -> bool:
        return id(w) in self.mods

    def add(self, x, w, y: torch.Tensor) -> torch.Tensor:
        """``y`` (the projection's bf16 or fp32 output, or a row-parallel rank's partial) plus this shard's term."""

        lora = self.mods.get(id(w))
        if lora is None:
            return y
        if isinstance(x, tuple):
            raise RuntimeError("an adapter needs bf16 prompt rows: start the server without --prefill-fp8")
        return (y.float() + lora(x)).to(y.dtype)


def _owner(layer, path: str):
    obj = layer
    *parents, leaf = path.split(".")
    for p in parents:
        obj = getattr(obj, p)
        if obj is None:
            return None, None
    return obj, leaf


def load(adapter_dir: str | Path, w, rank: int = 0, world: int = 1, *, name: str = "clef",
         device=None, volta: bool | None = None) -> Adapter:
    """Every manifest projection's factors for this rank's shard of ``w`` (the rank's ``Weights``)."""

    from safetensors import safe_open

    from .distributed import output_rows

    adapter_dir = Path(adapter_dir)
    man = json.loads((adapter_dir / "manifest.json").read_text())
    if volta is None:
        from tensorfold.cuda.build import volta as on_volta

        volta = on_volta()
    dev = device if device is not None else w.norm.device
    c = w.config
    full_cfg = c                                      # local configs divide heads; the split helpers take full sizes
    mods, owners = {}, []
    with safe_open(str(adapter_dir / "adapter.safetensors"), framework="pt") as f:
        for m in man["modules"]:
            path, kind, extra = _PROJ[m["proj"]]
            layer = w.layers[m["layer"]]
            parent, leaf = _owner(layer, path)
            proj = getattr(parent, leaf, None) if parent is not None else None
            if proj is None:
                fused = (getattr(getattr(layer, "gdn", None), "zba", None) is not None
                         or getattr(getattr(layer, "attn", None), "kv", None) is not None)
                raise ValueError(f"layer {m['layer']} has no {m['proj']} projection"
                                 + (" (stacked z|b|a or k|v projections take no adapter yet)" if fused else ""))
            A = f.get_tensor(m["key"] + ".lora_A").to(torch.bfloat16)      # [r, in]
            B = f.get_tensor(m["key"] + ".lora_B").to(torch.bfloat16)      # [out, r]
            out_full, in_full = int(m["out"]), int(m["in"])
            if world > 1:
                if kind == "col":
                    # split_output's rows: per segment, the rank's contiguous share (block only checks divisibility)
                    seg = extra(_FullSizes(full_cfg, world))
                    rows = output_rows(out_full, rank, world, segments=seg.get("segments"), block=seg.get("block", 1))
                    B = B.index_select(0, rows)
                else:
                    share = in_full // world
                    A = A[:, rank * share:(rank + 1) * share]
            n_local = getattr(proj, "n", None)
            k_local = getattr(proj, "k", None)
            if n_local is not None and n_local != B.shape[0]:
                raise ValueError(f"{m['key']}: adapter rows {B.shape[0]} != projection outputs {n_local}")
            if k_local is not None and k_local != A.shape[1]:
                raise ValueError(f"{m['key']}: adapter inputs {A.shape[1]} != projection inputs {k_local}")
            mods[id(proj)] = Lora(A.to(dev), B.to(dev), volta)
            owners.append(proj)
    return Adapter(name, mods, owners, man)


class _FullSizes:
    """The unsplit head counts and widths a rank-local config came from (split_layer divides them by the world)."""

    def __init__(self, c, world: int):
        self.k_heads, self.v_heads = c.k_heads * world, c.v_heads * world
        self.dk, self.dv, self.head_dim = c.dk, c.dv, c.head_dim


@contextlib.contextmanager
def applied(adapter: Adapter | None):
    """Prompt matmuls inside add ``adapter``'s terms (None: nothing changes)."""

    global ACTIVE
    if adapter is None:
        yield
        return
    with _LOCK:
        if ACTIVE is not None:
            raise RuntimeError("an adapter is already applied")
        ACTIVE = adapter
    try:
        yield
    finally:
        ACTIVE = None
