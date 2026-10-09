"""A prompt chunk on several ranks as two halves whose rank sums run beside the other half's layers.

Where the ranks' row-parallel sums go through host memory (no P2P), a chunk spends more time copying partials than
computing. Here the chunk's rows are two halves taken layer by layer, A then B: a half's sum runs on a second stream
while the GPU computes the other half, so the copies hide behind the matmuls and attention. B reads A's keys and
DeltaNet state of the same layer, which A wrote just before on the compute stream. The halves are what two chunks of
half the rows compute, and prompt bits never depend on chunking, so a chunk gives the bits ``prefill_chunk`` gives.
``TF_PROMPT_OVERLAP=0`` keeps one pass a chunk.
"""

from __future__ import annotations

import os

import torch

from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.cuda.kernels.prefill_attention import attention

from . import glue
from .forward import grow as _grow

MIN_ROWS = 1024           # a chunk this short runs in one pass (its halves would not fill the GPU)
_streams: dict = {}


def enabled() -> bool:
    return os.environ.get("TF_PROMPT_OVERLAP", "1") != "0"


def usable(w, tp: bool, rows: int, cut: int, vision) -> bool:
    """Whether a chunk takes the two halves: several ranks summing through host memory, a dense text chunk, no cut."""

    if not (enabled() and tp and rows >= MIN_ROWS and not cut and vision is None):
        return False
    if any(getattr(layer, "moe", None) is not None for layer in w.layers):
        return False
    from tensorfold.cuda import p2p

    peer = p2p.peer()
    return peer is not None and hasattr(peer, "reduce")


def _comm(device, j: int = 0) -> torch.cuda.Stream:
    s = _streams.get((device, j))
    if s is None:
        s = _streams[(device, j)] = torch.cuda.Stream(device=device)
    return s


def two_sums() -> bool:
    """``TF_PROMPT_TWO_SUMS``: each half's sums on its own stream and host buffers (default), or both on one."""

    return os.environ.get("TF_PROMPT_TWO_SUMS", "1") != "0"


class _Sum:
    """One row-parallel sum on a side stream, after the partial is ready; ``result`` makes the compute stream wait.
    With ``peer`` (a half's own host-memory sum), the halves' sums run on two streams and their copies overlap."""

    def __init__(self, partial: torch.Tensor, comm: torch.cuda.Stream, peer=None) -> None:
        from .distributed import gather_rank_partials

        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(comm):
            comm.wait_event(ready)
            if peer is not None and partial.dtype == torch.bfloat16 and peer.fits(partial):
                self.out = peer.reduce(partial)     # the same bits as gather_rank_partials
            else:
                self.out = gather_rank_partials(partial)
            partial.record_stream(comm)
            self.done = torch.cuda.Event()
            self.done.record(comm)

    def result(self) -> torch.Tensor:
        torch.cuda.current_stream().wait_event(self.done)
        self.out.record_stream(torch.cuda.current_stream())
        return self.out


def _half(w, pg, mm_group, mm, ids: torch.Tensor, st, p0: int, a: int, b: int, capture_taps: bool):
    """A half's layers as a generator: it yields each row-parallel partial and takes back its sum; last, its final
    (x, pending, taps)."""

    c = w.config
    W = b - a
    keep = c.conv_kernel - 1
    dev = ids.device
    pos = torch.arange(p0 + a, p0 + b, device=dev, dtype=torch.int32)
    windows = (torch.arange(W, device=dev, dtype=torch.int32)[:, None]
               + torch.arange(keep + 1, device=dev, dtype=torch.int32)[None, :])
    x = glue.embedding(ids[a:b].to(torch.int32), w.embed)
    pending = None
    taps = []
    from .prefill import _read, _stored

    for i, layer in enumerate(w.layers):
        x, h = pg.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            if gdn.zba is not None:
                qkv, zba = mm_group(h, [gdn.qkv, gdn.zba])
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                bb = zba[:, vd:vd + c.v_heads].contiguous()
                aa = zba[:, vd + c.v_heads:].contiguous()
            else:
                qkv, z, bb, aa = mm_group(h, [gdn.qkv, gdn.z, gdn.b, gdn.a])
                z = z.reshape(W, c.v_heads, c.dv)
            q, k, v, g, beta = glue.gdn_pre(qkv, st.conv[i], gdn.conv, windows, aa, bb, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk)
            final = torch.empty_like(st.rec[i])
            yr = deltanet.chain(q, k, v, g, beta, st.rec[i], final)
            st.conv[i] = torch.cat([st.conv[i], qkv[-keep:]])[-keep:].contiguous()
            st.rec[i] = final
            part = mm(pg.gated_norm(yr, z, gdn.norm, c.eps), gdn.out)
        else:
            attn = layer.attn
            if attn.kv is not None:
                qg, kv = mm_group(h, [attn.q, attn.kv])
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                qg, key, value = mm_group(h, [attn.q, attn.k, attn.v])
                value = value.reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            kbuf, vbuf = _grow(st, i, p0 + b)
            kbuf[p0 + a:p0 + b], vbuf[p0 + a:p0 + b] = (_stored(w, key.view(W, c.kv_heads, c.head_dim)),
                                                         _stored(w, value))
            out = attention(q.view(W, c.heads, c.head_dim), _read(w, kbuf, p0 + b), _read(w, vbuf, p0 + b), p0 + a,
                            scale=c.head_dim ** -0.5)
            part = mm(pg.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim), attn.o)
        r = yield part
        x, h = pg.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = yield mm(pg.swiglu(*mm_group(h, [layer.gate, layer.up])), layer.down)
        if capture_taps and i in w.tap_layers:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    yield x, pending, taps


@torch.no_grad()
def prefill_chunk_overlap(w, tokens: torch.Tensor, st, *, capture_taps: bool = False, last: bool = True,
                          every: bool = False):
    """``prefill_chunk(..., tp=True)`` of a text chunk as two halves, their sums beside each other's layers."""

    from . import prefill_bf16, prefill_glue
    from .prefill import _mm, _mm_group
    from tensorfold.cuda import prompt_precision

    c = w.config
    pg = prefill_glue if w.fast_prefill and prompt_precision.fp8() else prefill_bf16
    W, p0 = int(tokens.shape[0]), st.pos
    mid = W // 2
    from tensorfold.cuda import p2p

    two = two_sums()
    peers = [p2p.peer(), p2p.second()] if two else [None, None]
    two = two and peers[1] is not None
    comms = [_comm(tokens.device, 0), _comm(tokens.device, 1 if two else 0)]
    for comm in comms:
        comm.wait_stream(torch.cuda.current_stream())
    halves = [_half(w, pg, _mm_group, _mm, tokens, st, p0, 0, mid, capture_taps),
              _half(w, pg, _mm_group, _mm, tokens, st, p0, mid, W, capture_taps)]
    # A's first partial, then B's (B reads A's layer-0 state)
    sums = [_Sum(next(g), comms[j], peers[j] if two else None) for j, g in enumerate(halves)]
    ends = [None, None]
    while ends[1] is None:
        for j, g in enumerate(halves):
            got = g.send(sums[j].result())
            if isinstance(got, tuple):
                ends[j] = got
            else:
                sums[j] = _Sum(got, comms[j], peers[j] if two else None)
    for comm in comms:
        torch.cuda.current_stream().wait_stream(comm)
    st.pos = p0 + W
    (xa, pa, ta), (xb, pb, tb) = ends
    normed = None
    if every:
        _, normed, _ = glue.add_rmsnorm(torch.cat([xa, xb]), torch.cat([pa, pb]), w.norm, c.eps)
    elif last:
        _, normed, _ = glue.add_rmsnorm(xb[-1:].contiguous(), pb[-1:].contiguous(), w.norm, c.eps)
    taps_out = torch.cat([torch.cat([u, v]) for u, v in zip(ta, tb)], dim=-1) if capture_taps else None
    return normed, taps_out
