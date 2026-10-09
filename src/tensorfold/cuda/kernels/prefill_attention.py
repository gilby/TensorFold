"""Prefill attention in 64-key tiles by absolute position, so chunking never changes bits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from tensorfold.cuda.build import volta
from tensorfold.families.qwen4_exp.cuda.kvquant import h32

BM = 64
BN = 64


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    s = tl.where(valid, s, float("-inf"))
    tile_m = tl.max(s, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _attend(Q, K, V, OUT, p0, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, SCALE: tl.constexpr):
    block = tl.program_id(0)
    head = tl.program_id(1)
    hk = head // (H // HK)
    rows = block * BM + tl.arange(0, BM)
    ok = rows < W
    pos = p0 + rows
    d = tl.arange(0, D)
    q = tl.load(Q + (rows[:, None] * H + head) * D + d[None, :], mask=ok[:, None], other=0.0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    first_pos = p0 + block * BM
    last_pos = p0 + tl.minimum(block * BM + BM, W) - 1
    full = (first_pos + 1) // BN                  # tiles every row of the block sees whole
    for t in range(0, full):
        keys = t * BN + tl.arange(0, BN)
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :])
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :])
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        seen = keys <= last_pos
        k = tl.load(K + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        v = tl.load(V + (keys[:, None] * HK + hk) * D + d[None, :], mask=seen[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    out = o / l[:, None]
    tl.store(OUT + (rows[:, None] * H + head) * D + d[None, :], out.to(tl.bfloat16), mask=ok[:, None])


def attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float) -> torch.Tensor:
    """q (W, H, D) bf16 at positions [p0, p0 + W); the caches must already hold every key through p0 + W - 1."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    if volta(q.device):
        if prompt_mode() == "fused":
            return fused_attention(q, k_cache, v_cache, p0, scale=scale)
        if k_cache.dtype == torch.int8:
            raise ValueError("packed int8 caches take the fused sm_70 prompt kernel (TF_PROMPT_ATTN=fused)")
        if prompt_mode() == "blocked16":
            return blocked_attention(q, k_cache, v_cache, p0, scale=scale, hk=hk, half_scores=True)
        if blocked_on():
            return blocked_attention(q, k_cache, v_cache, p0, scale=scale, hk=hk)
        return volta_attention(q, k_cache, v_cache, p0, scale=scale, hk=hk)
    out = torch.empty_like(q)
    if d == 64:
        return triton_attention(q, k_cache, v_cache, p0, scale=scale, out=out)
    _ext().prefill_attention(q, k_cache, v_cache, out, p0, scale, heads_a_block(h // hk))
    return out


def triton_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """The same through ``_attend``: the bits' definition, one program a 64-row block and query head."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    out = torch.empty_like(q) if out is None else out
    _attend[(triton.cdiv(w, BM), h)](q, k_cache, v_cache, out, p0, w, H=h, HK=hk, D=d, BM=BM, BN=BN, SCALE=scale,
                                     num_warps=8, num_stages=1 if d > 128 else 2)
    return out


def heads_a_block(group: int) -> int:
    """Query heads of one KV head in a CUDA block of eight warps: the largest of 8, 4, 2, 1 dividing the group."""

    return next(n for n in (8, 4, 2, 1) if group % n == 0)


def _check(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int) -> int:
    w, h, d = q.shape
    hk = k_cache.shape[1]
    if k_cache.shape[0] < p0 + w or v_cache.shape != k_cache.shape or h % hk or d not in (64, 128, 256):
        raise ValueError("prefill attention: caches must hold the chunk's keys; heads a multiple of kv heads")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("prefill attention takes contiguous tensors")
    return hk


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_prefill_attention_v1", sources=[str(here / "prefill_attention.cpp"),
                                                                 str(here / "prefill_attention.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


VOLTA_ALIGN = 128         # query rows pad back to a multiple of this: a row's tile in the kernel never moves with chunking


def volta_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
                    hk: int) -> torch.Tensor:
    """sm_70: PyTorch's memory-efficient (CUTLASS) attention in fp16 with a bottom-right causal mask, one key head at
    a time (its query heads on the batch axis, its keys and values broadcast to them without a copy); the bf16
    kernels need sm_80. Chunk invariant through ``VOLTA_ALIGN``, not the Triton tiles' bits. The fp16 keys and
    values of one head are the pass's largest transient, which the engine's geometry counts on sm_70."""

    w, h, d = q.shape
    n, g = p0 + w, h // hk
    lead = p0 % VOLTA_ALIGN
    qh = q.to(torch.half)
    if lead:
        qh = torch.cat([qh.new_zeros((lead, h, d)), qh])
    qb = qh.view(lead + w, hk, g, d)
    out = torch.empty((lead + w, hk, g, d), dtype=torch.half, device=q.device)
    for j in range(hk):
        k, v = k_cache[:n, j].to(torch.half), v_cache[:n, j].to(torch.half)
        qj = qb[:, j].permute(1, 0, 2)[:, :, None]                      # (g, rows, 1, d): the head's queries
        kj, vj = k[None, :, None].expand(g, n, 1, d), v[None, :, None].expand(g, n, 1, d)
        o = torch.ops.aten._efficient_attention_forward(qj, kj, vj, None, None, None, lead + w, n, 0.0, 2, False,
                                                        scale=scale)[0]
        out[:, j] = o[:, :, 0].permute(1, 0, 2)
    return out[lead:].reshape(w, h, d).to(torch.bfloat16)


# ---- sm_70: attention as tensor-core GEMMs over absolute key blocks ---------------------------------------------

KEY_BLOCK = 2048          # keys a block: absolute [j * KEY_BLOCK, (j + 1) * KEY_BLOCK), always whole (zeros past the keys)


def prompt_mode() -> str:
    """``TF_PROMPT_ATTN`` on sm_70: ``blocked16`` (the default: cuBLASLt GEMMs over absolute key blocks, the scores
    scaled and rounded to fp16 in the GEMM's epilogue), ``blocked`` (fp32 scores), ``fused`` (one causal mma.m8n8k4
    pass, nothing to DRAM, int8 caches read packed; measured slower) or ``memeff`` (PyTorch's CUTLASS kernel)."""

    import os

    return os.environ.get("TF_PROMPT_ATTN", "blocked16")


def prompt_v4() -> bool:
    """The fused prompt kernel's tile: v4 (values coalesced and row-major, fp16 dequant) unless TF_DECODE_ATTN=v2."""

    from .attention import decode_v4

    return decode_v4()


def blocked_on() -> bool:
    return prompt_mode() != "memeff"


def reads_packed(device) -> bool:
    """Whether a prompt's attention reads int8 caches as stored (no unpack to bf16 first)."""

    return volta(device) and prompt_mode() == "fused"


@triton.jit
def _rotate_back(X, OUT, n, D: tl.constexpr, R: tl.constexpr):
    """R rows of D fp32 values in the cache's H32 rotation -> the model's basis (H32 is its own inverse), bf16."""

    rows = tl.program_id(0) * R + tl.arange(0, R)
    ok = rows < n
    d = tl.arange(0, D)
    x = tl.load(X + rows[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
    x = h32(tl.reshape(x, (R * D // 32, 32)), M=R * D // 32)
    tl.store(OUT + rows[:, None] * D + d[None, :], tl.reshape(x, (R, D)).to(tl.bfloat16), mask=ok[:, None])


def fused_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *,
                    scale: float) -> torch.Tensor:
    """sm_70 causal prompt attention in one kernel (attention_volta.cu ``prompt``): bf16 caches, or int8 rows as packed
    by kvpack (the query rides their rotation, the fp32 output is rotated back and rounded once)."""

    from . import kvpack
    from .attention import _volta_ext

    w, h, d = q.shape
    packed = k_cache.dtype == torch.int8
    qq = kvpack.rotate(q) if packed else q.contiguous()
    out = torch.empty((w, h, d), dtype=torch.float32, device=q.device)
    _volta_ext().prompt(qq, k_cache.contiguous(), v_cache.contiguous(), out, p0, scale, prompt_v4())
    if not packed:
        return out.to(torch.bfloat16)
    res = torch.empty((w, h, d), dtype=torch.bfloat16, device=q.device)
    rows = w * h
    _rotate_back[(triton.cdiv(rows, 4),)](out, res, rows, D=d, R=4, num_warps=4)
    return res


@triton.jit
def _block_softmax(S, P, M, L, A, p0, k0, G: tl.constexpr, B: tl.constexpr, SCALE: tl.constexpr):
    """One query row against a key block: scores scaled and masked by position, the running max and sum updated,
    exp(s - max) as fp16 for the PV GEMM, and the factor the row's earlier output is rescaled by."""

    r = tl.program_id(0)
    cols = tl.arange(0, B)
    pos = p0 + r // G
    valid = (k0 + cols) <= pos
    s = tl.load(S + r.to(tl.int64) * B + cols) * SCALE
    s = tl.where(valid, s, float("-inf"))
    m = tl.load(M + r)
    top = tl.max(s, 0)
    nm = tl.maximum(m, top)
    seen = nm != float("-inf")
    alpha = tl.where(top == float("-inf"), 1.0, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)))
    p = tl.where(valid & seen, tl.exp(s - nm), 0.0)
    tl.store(P + r.to(tl.int64) * B + cols, p.to(tl.float16))
    tl.store(M + r, nm)
    tl.store(L + r, tl.load(L + r) * alpha + tl.sum(p, 0))
    tl.store(A + r, alpha)


@triton.jit
def _block_softmax16(S, P, M, L, A, p0, k0, G: tl.constexpr, B: tl.constexpr):
    """``_block_softmax`` on fp16 scores the GEMM already scaled: one read of the scores, one write of P."""

    r = tl.program_id(0)
    cols = tl.arange(0, B)
    pos = p0 + r // G
    valid = (k0 + cols) <= pos
    s = tl.load(S + r.to(tl.int64) * B + cols).to(tl.float32)
    s = tl.where(valid, s, float("-inf"))
    m = tl.load(M + r)
    top = tl.max(s, 0)
    nm = tl.maximum(m, top)
    seen = nm != float("-inf")
    alpha = tl.where(top == float("-inf"), 1.0, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)))
    p = tl.where(valid & seen, tl.exp(s - nm), 0.0)
    tl.store(P + r.to(tl.int64) * B + cols, p.to(tl.float16))
    tl.store(M + r, nm)
    tl.store(L + r, tl.load(L + r) * alpha + tl.sum(p, 0))
    tl.store(A + r, alpha)


@lru_cache(maxsize=1)
def _gemm16():
    from tensorfold.cuda.kernels import qmm_volta

    return qmm_volta._ext().gemm16


@lru_cache(maxsize=1)
def _gemm():
    from tensorfold.cuda.kernels import qmm_volta

    return qmm_volta._ext().gemm


def blocked_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
                      hk: int, half_scores: bool = False) -> torch.Tensor:
    """sm_70 prompt attention as fp16 tensor-core GEMMs (the prompt matmuls' pinned cuBLASLt plans, fp32 sums): a key
    head's query heads as rows, each absolute key block's scores, an fp32 online softmax a row, then P.V. A row's
    blocks and the order they fold in are fixed by its position, a block a row sees nothing of changes none of its bits,
    and the GEMMs' rows are independent of the row count, so chunking never changes bits."""

    w, h, d = q.shape
    n, g = p0 + w, h // hk
    gemm = _gemm()
    blocks = -(-n // KEY_BLOCK)
    out = torch.empty((w, h, d), dtype=torch.bfloat16, device=q.device)
    rows = w * g
    for j in range(hk):
        qj = q[:, j * g:(j + 1) * g].reshape(rows, d).to(torch.half).contiguous()
        k16 = torch.zeros((blocks * KEY_BLOCK, d), dtype=torch.half, device=q.device)
        k16[:n] = k_cache[:n, j]
        vt = torch.zeros((d, blocks * KEY_BLOCK), dtype=torch.half, device=q.device)
        vt[:, :n] = v_cache[:n, j].t()
        m = torch.full((rows,), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((rows,), dtype=torch.float32, device=q.device)
        a = torch.empty_like(l)
        o = torch.zeros((rows, d), dtype=torch.float32, device=q.device)
        p = torch.empty((rows, KEY_BLOCK), dtype=torch.half, device=q.device)
        for b in range(blocks):
            k0 = b * KEY_BLOCK
            if half_scores:                  # scaled in the GEMM's epilogue, rounded once to fp16: half the bytes
                sc = _gemm16()(qj, k16[k0:k0 + KEY_BLOCK], scale)
                _block_softmax16[(rows,)](sc, p, m, l, a, p0, k0, G=g, B=KEY_BLOCK, num_warps=8)
            else:
                sc = gemm(qj, k16[k0:k0 + KEY_BLOCK])
                _block_softmax[(rows,)](sc, p, m, l, a, p0, k0, G=g, B=KEY_BLOCK, SCALE=scale, num_warps=8)
            del sc
            o.mul_(a[:, None]).add_(gemm(p, vt[:, k0:k0 + KEY_BLOCK].contiguous()))
        out[:, j * g:(j + 1) * g] = (o / l[:, None]).view(w, g, d).to(torch.bfloat16)
    return out
