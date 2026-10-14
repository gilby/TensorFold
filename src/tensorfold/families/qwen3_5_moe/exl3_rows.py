"""Qwen3.6 MoE EXL3 decode rows: row_forward's walk and records, the layers' matmuls in fused EXL3 launches."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.exl3 import experts, glue, rows, tiles
from tensorfold.kernels.qwen.dense.v1 import row_forward
from tensorfold.kernels.qwen.dense.v1.row_forward import _attend, _conv_windows, _recur, _Rows, _token_ids

MAX_ROWS = 128           # rows a forward takes (every row's bits are its one-row bits at any count)


def _attention_rows(attn: Any, qkv: mx.array, items: Sequence[Any], rows_: _Rows) -> mx.array:
    """row_forward._attention from the stacked q|k|v rows [R, Nq + Nk + Nv]: attention's output [1, R, H * D] (its gate
    is applied with o_proj's rotation)."""

    R = int(qkv.shape[0])
    B, L = 1, R
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    sizes = attn.q_proj.stack.sizes
    nq, nk = sizes[0], sizes[1]
    q_out, k_out, v_out = qkv[None, :, :nq], qkv[None, :, nq:nq + nk], qkv[None, :, nq + nk:]
    D = int(attn.head_dim) if hasattr(attn, "head_dim") else nq // (2 * H)
    queries = attn.q_norm(q_out.reshape(B, L, 2 * H, D))[:, :, 0::2]
    keys = attn.k_norm(k_out.reshape(B, L, nkv, -1))
    values = v_out.reshape(B, L, nkv, -1)
    queries, keys, values = (t.transpose(0, 2, 1, 3) for t in (queries, keys, values))
    pos = rows_.positions
    if any(getattr(c, "vision_rope_delta", 0) for c in items):
        shifts = [int(getattr(c, "vision_rope_delta", 0)) for c, w in zip(items, rows_.widths) for _ in range(w)]
        pos = pos + mx.array(shifts, dtype=mx.int32)
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    if rows_.single:
        out = _attend(attn, queries, keys, values, items[0], rows_.parents[0], rows_.chains[0], rows_.records[0])
    else:
        outs = []
        for s, item in enumerate(items):
            a, w = rows_.offsets[s], rows_.widths[s]
            outs.append(_attend(attn, queries[:, :, a:a + w], keys[:, :, a:a + w], values[:, :, a:a + w], item,
                                rows_.parents[s], rows_.chains[s], rows_.records[s]))
        out = mx.concatenate(outs, axis=2)
    return out.transpose(0, 2, 1, 3).reshape(B, L, -1)


def _gdn_rows(gdn: Any, y: mx.array, items: Sequence[Any], rows_: _Rows) -> mx.array:
    """row_forward._gdn from the stacked [qkv | z | b | a] rows: out_proj's rotated input XH [1, R, NV * DV]."""

    from tensorfold.kernels.qwen.dense.v1 import row_streams

    R = int(y.shape[0])
    y = y.reshape(1, R, -1)
    n_keep = gdn.conv_kernel_size - 1
    out_suh = gdn.out_proj.stack.suh
    if rows_.single:
        if rows_.windows is None:
            rows_.windows = [_conv_windows(p, n_keep) for p in rows_.parents]
        rec = _recur(gdn, y, items[0], rows_.parents[0], rows_.chains[0], rows_.windows[0], rows_.records[0], n_keep)
        return glue.gdn_post_rotate(rec, y, gdn.norm.weight, gdn.norm.eps, out_suh, zo=gdn.conv_dim)
    recs = []
    for g0 in range(0, len(items), row_streams.GROUP):
        g1 = min(g0 + row_streams.GROUP, len(items))
        a, b = rows_.offsets[g0], rows_.offsets[g1 - 1] + rows_.widths[g1 - 1]
        chain = all(rows_.chains[g0:g1])
        rec, per = row_streams.recur(gdn, y[:, a:b], items[g0:g1], rows_.parents[g0:g1], chain, n_keep)
        for record, (q, k, v, g, beta, state, conv_state, ys, state_out, tails) in zip(rows_.records[g0:g1], per):
            record.append(("gdn", q, k, v, g, beta, state, (conv_state, ys, gdn.conv_dim, state_out, tails, chain),
                           n_keep))
        recs.append(rec)
    rec = recs[0] if len(recs) == 1 else mx.concatenate(recs, axis=1)
    return glue.gdn_post_rotate(rec, y, gdn.norm.weight, gdn.norm.eps, out_suh, zo=gdn.conv_dim)


def _in_suh(inner: Any) -> mx.array:
    return (inner.linear_attn.in_proj_qkv if getattr(inner, "is_linear", False) else inner.self_attn.q_proj).stack.suh


def rows_forward(core: Any, windows: Sequence[Any], parents: Sequence[Sequence[int]], caches: Sequence[list[Any]],
                 starts: Sequence[int], *, pipeline_layers: int = 4, first_alone: bool = True
                 ) -> tuple[mx.array, _Rows]:
    """Final-normed rows [1, R, D] and stream records, as row_forward._rows_forward: a layer is a norm reading the last
    block's partial outputs, one projection launch, attention or the recurrence, the output projection, a norm with the
    router's logits, the route and the two expert launches."""

    widths = [len(p) for p in parents]
    if len(windows) != len(parents) or len(caches) != len(parents) or len(starts) != len(parents):
        raise ValueError(f"exl3 rows: {len(windows)} windows, {len(parents)} parent lists, {len(caches)} caches, "
                         f"{len(starts)} starts")
    if sum(widths) > MAX_ROWS:
        raise ValueError(f"exl3 rows: {sum(widths)} rows, a forward takes up to {MAX_ROWS}")
    rows_ = _Rows(parents, starts)
    hidden = core.embed_tokens(_token_ids(windows))[0]
    R = int(hidden.shape[0])
    layers = list(core.layers)
    ye = zp = None
    tapped: Any = None
    for index, (layer, *items) in enumerate(zip(layers, *caches)):
        inner = getattr(layer, "_layer", layer)
        ln = inner.input_layernorm
        hidden, x, xh = glue.norm_rotate(hidden, ln.weight, ln.eps, ye=ye, zp=zp, suh=_in_suh(inner))
        if tapped is not None:
            tapped[0][tapped[1]] = hidden[None]
        if getattr(inner, "is_linear", False):
            gdn = inner.linear_attn
            y = rows.project(gdn.in_proj_qkv.stack, xh, plain=(x, gdn._exl3_ba))
            oh = _gdn_rows(gdn, y, items, rows_)[0].reshape(1, R, -1)
            out = gdn.out_proj.stack
        else:
            attn = inner.self_attn
            qkv = rows.project(attn.q_proj.stack, xh)
            o_in = _attention_rows(attn, qkv, items, rows_)[0]
            out = attn.o_proj.stack
            head_dim = int(getattr(attn, "head_dim", 0)) or attn.q_proj.stack.sizes[0] // (2 * attn.num_attention_heads)
            oh = tiles.rotate_gated(o_in, qkv, head_dim, out.suh)
        z = rows.project_split(out, oh)
        ex = inner.mlp.experts
        pl = inner.post_attention_layernorm
        hidden, x, logits = glue.norm_router(hidden, pl.weight, pl.eps, ex.router, zp=(z, out.svh))
        gu, slots, weights, units = experts.route_gate_up(x, logits, ex)
        ye = experts.down(gu, slots, weights, units, ex)
        zp = None
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) \
                and index + 1 < len(layers):
            mx.async_eval(hidden, ye)
    hidden, x = glue.norm_rotate(hidden, core.norm.weight, core.norm.eps, ye=ye)[:2]
    if tapped is not None:
        tapped[0][tapped[1]] = hidden[None]
    return x[None], rows_


def install(core: Any) -> None:
    """Route row_forward's walk for this core through ``rows_forward``."""

    core._tensorfold_rows_forward = rows_forward


__all__ = ["MAX_ROWS", "install", "rows_forward"]
