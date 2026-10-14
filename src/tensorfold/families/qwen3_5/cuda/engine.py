"""The Qwen3.8-27B CUDA engine: one GPU, two ranks or four; prompt ends are kept, replies are prefilled again."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import os

from tensorfold.cuda import prompt_precision
from tensorfold.cuda.build import VOLTA, volta


KEEP = 3             # prompt states a concurrent decoder keeps to resume from (each holds a DeltaNet copy)
KEEP_ONE = 4         # prompt states one stream keeps (they share its attention buffers)


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's cache entry ends: one token early, since a next turn sent back without its reasoning renders ``<think>`` and two newlines there."""

    return max(1, len(prompt) - 1)


class Qwen27Engine:
    """Qwen3.8-27B on one GPU or several ranks (rank 0 here), DFlash2 drafting, prefix reuse."""

    tree_rows: int | None = None       # a lone stream's tree rows on one GPU (None: max_rows, as in 0.5.0)
    room = None                        # one GPU's attention-cache budget (streams.KVRoom), None on two

    def __init__(self, model_dir: Path, draft_dir: Path | None, *, max_rows: int = 12, tp: int = 1,
                 rank: int = 0, master: str = "", port: int = 29551, split_head: bool = False,
                 tp_draft: bool = False, allow_copy: bool = True, streams: int = 1,
                 context: int | None = None, context_explicit: bool | None = None, vision: bool = False,
                 vision_urls: bool = False, vision_offload: bool = False, tree_rows: int | None = None,
                 keep: int | None = None, loader=None, kv_dtype: str = "bf16"):
        import torch

        from tensorfold.cuda.nvfp4.format import is_quantized

        from .exl3_load import admission, quant_config

        exl3 = quant_config(Path(model_dir)) is not None
        nvfp4 = not exl3 and is_quantized(Path(model_dir))
        if tp not in (1, 2, 4) or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Qwen3.8-27B runs on one GPU, two ranks or four")
        if exl3 and tp != 1:
            raise ValueError("EXL3 packs of Qwen3.8-27B run on one GPU: drop --tp 2, or serve the MLX checkpoint "
                             "(TensorFold/Qwen3.8-27B-MLX-4bit) or an NVFP4 one on two")
        if exl3 and volta():
            raise ValueError("EXL3 packs of Qwen3.8-27B need sm_80 or newer kernels: on a Volta GPU serve an NVFP4 "
                             "checkpoint (RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) or the MLX one")
        from tensorfold.cuda.kernels.kvpack import check as check_kv

        self.kv_dtype = check_kv(kv_dtype)
        bits = 8 if self.kv_dtype == "int8" else 16
        if nvfp4 and vision:
            raise ValueError("image input on CUDA is tested on the MLX checkpoint only: drop --vision for an NVFP4 "
                             "checkpoint, or serve TensorFold/Qwen3.8-27B-MLX-4bit")
        from .weights import load
        from tensorfold.cuda.capacity import admit, config, gather_ints, total_bytes
        from tensorfold.cuda.geometry import (draft_geometry, gdn_geometry, live_kv, prompt_row_bytes, prompt_rows,
                                              stream_geometry)
        from .affine_memory import draft_weights, weight_transform
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights

        self.torch = torch
        self.tp, self.rank, self.max_rows, self.allow_copy = tp, rank, max_rows, allow_copy
        self.tree_rows = None if tree_rows is None else min(int(tree_rows), max_rows)
        self.vision = None
        self.vision_enabled = bool(vision)
        __import__("tensorfold.cuda.device", fromlist=["select"]).select(torch)
        if streams > 1 and tp == 1:          # streams' caches of many sizes come and go: growable segments, less slack
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        keep = KEEP if keep is None else int(keep)         # --checkpoint-slots: each kept state is in the estimate
        from .tree_graphs import enabled as tree_graphs_enabled

        # tree verify graphs: no EXL3 weights; on sm_70 several ranks only (where the host bounds a round): a lone
        # stream's rounds, and with --parallel several streams' rounds too (RoundGraphs, TF_ROUND_GRAPHS)
        trees_on = tree_graphs_enabled() and not exl3 and (not volta() or tp > 1)
        round_graphs = os.environ.get("TF_ROUND_GRAPHS", "1")
        rounds_on = trees_on and volta() and tp > 1 and streams > 1 and round_graphs != "0"
        if tp > 1:
            import torch.distributed as dist

            from .distributed import split_weights

            dist.init_process_group("nccl", init_method=f"tcp://{master}:{port}", rank=rank, world_size=tp)
            # both ranks must run the same calls: refuse to start when they were given different settings
            flags = torch.tensor([int(draft_dir is not None and tp_draft), max_rows, int(split_head), int(allow_copy),
                                  streams, -1 if context is None else int(context), int(bool(context_explicit)),
                                  int(vision), keep, int(trees_on), int(rounds_on) + int(round_graphs == "force"),
                                  bits, int(prompt_precision.fp8())],   # precision last
                                 dtype=torch.int64, device="cuda")
            both = torch.empty((tp, flags.numel()), dtype=torch.int64, device="cuda")
            dist.all_gather_into_tensor(both, flags)
            for r in range(1, tp):
                prompt_precision.same_on_ranks(int(both[0, -1]), int(both[r, -1]))
            odd = next((r for r in range(1, tp) if not torch.equal(both[0], both[r])), None)
            if odd is not None:
                raise RuntimeError("the ranks were started with different settings (two-rank drafter, rows, "
                                   f"head split, copies, --parallel, --context, --checkpoint-slots): rank 0 "
                                   f"{both[0].tolist()}, rank {odd} {both[odd].tolist()}; pull the draft model "
                                   "everywhere, and pass the same --no-drafts, --parallel, --context, "
                                   "--checkpoint-slots, --kv-dtype, TF_TREE_GRAPHS and TF_ROUND_GRAPHS to every rank")
            gather = lambda values: gather_ints(torch, lambda send, recv: dist.all_gather_into_tensor(recv, send), values,
                                                tp)
            from tensorfold.cuda import p2p

            # row-parallel sums in one launch (P2P, or shared host memory); prompt chunks' bf16 partials by copies
            stage = max(p2p.STAGE_BYTES, 4096 * int(config(model_dir)["hidden_size"]) * 2)
            if p2p.install(rank, stage_bytes=stage) and rank == 0:
                print(f"[tensorfold] rank sums: {p2p.peer().kind}", flush=True)
            elif rank == 0:
                print("[tensorfold] rank sums: NCCL", flush=True)
        else:
            gather = None
        many = streams > 1
        # prompt chunks sized to the card (4096 rows from 80 GB), the verify scratch to the rows a round takes
        chunk = prompt_rows(total_bytes(torch), prompt_row_bytes(config(model_dir), tp))
        sized = dict(prompt_staging=volta(), kv_bits=bits)
        # several streams grow by use (one GPU or several ranks): one whole window and each stream's first rows
        geometry = ((lambda text: stream_geometry(text, tp, streams, keep, first=256, **sized)) if many
                    else (lambda text: gdn_geometry(text, tp, max_rows, rows=max_rows, prompt=chunk, evicts=tp == 1, **sized)))
        # an affine checkpoint's packed words at their stored precision; an EXL3 pack's by its own format
        tensor_bytes = weight_transform(model_dir, one_gpu=tp == 1)
        if exl3:
            from .prefill import LAYER_MAJOR

            geometry, tensor_bytes = admission(geometry, layer_major=LAYER_MAJOR)
        elif nvfp4:
            from .nvfp4_load import admission as nvfp4_admission

            geometry, tensor_bytes = nvfp4_admission(geometry)
            if tp > 1:                      # each rank keeps its shard of every layer as it loads (no startup copy)
                from .distributed import half_layers

                tensor_bytes = half_layers(tensor_bytes, tp)
        # one admission for one stream or many, on every rank, before any weight loads
        self.capacity_plan = admit(model_dir, context, context_explicit, torch,
                                   capacity_geometry(geometry, model_dir, vision, rank, offload=vision_offload),
                                   vision_weights(tensor_bytes, vision, rank, vision_offload),
                                   rank=rank, world=tp, gather=gather,
                                   draft_dir=draft_dir if rank == 0 or tp_draft else None,
                                   draft_weights=draft_weights,
                                   draft_geometry=lambda text: draft_geometry(text, tp if tp_draft else 1, max_rows,
                                                                              bounded=True, streams=streams,
                                                                              kept=keep + 1 if many else 0),
                                   # sm_70 (32 GB V100s): the full checkpoint (already the resident term) is split and freed
                                   # before caches exist; a second full copy is not part of that peak; NVFP4 layers
                                   # shard as they load on two ranks, so they never hold a second full copy either
                                   startup_copies=int(tp > 1 and not nvfp4 and not volta()),
                                   # MLX and NVFP4 checkpoints run on the sm_70 kernels; EXL3 needs the shared ones
                                   need=None if exl3 or (nvfp4 and not volta()) else VOLTA,
                                   # sm_70 NVFP4: each projection repacks from its upload (the bf16 head a chunk at a
                                   # time), one extra copy of the largest tensor at most
                                   staging_copies=2 if nvfp4 and volta() else 3)
        self.context_window = self.capacity_plan["context_window"]
        if tp > 1 and nvfp4:                # layers shard as they load; the embedding and head stay whole for the drafter
            from .distributed import shard_layer

            full = load(model_dir, nvfp4_layer=shard_layer(rank, tp))
            self.w = split_weights(full, rank, tp, tiled=True, split_head=split_head, layers_split=True)
        elif tp > 1:
            full = load(model_dir)
            self.w = split_weights(full, rank, tp, tiled=True, split_head=split_head)
        else:
            full = loader(model_dir) if loader is not None else load(model_dir, tiled=True)
            self.w = full
        self.w.prompt_rows = chunk
        self.w.kv_dtype = self.kv_dtype
        if self.kv_dtype != "bf16" and rank == 0:
            print(f"[tensorfold] {self.kv_dtype} KV cache (fp16 scale per 32 values): "
                  f"{self.context_window} tokens a window", flush=True)
        self.draft = None
        if draft_dir is not None and (rank == 0 or (tp > 1 and tp_draft)):
            from .dflash1 import drafter_class

            # tp_draft: every rank holds its share of the drafter and they draft together (each needs --draft)
            kind = drafter_class(draft_dir)
            self.draft = kind(draft_dir, full, rank=rank, world=tp) if tp > 1 and tp_draft else kind(draft_dir, full)
            if tp > 1 and not tp_draft and hasattr(self.draft, "own_head"):
                self.draft.own_head()           # rank 0 alone drafts: the whole target head it was given goes
            if len(self.draft.taps) and max(self.draft.taps) >= full.config.layers:
                raise ValueError(f"the drafter reads target layers {self.draft.taps}; this model has {full.config.layers}")
            self.w.tap_layers = full.tap_layers = self.draft.taps
        # one stream with a chain drafter (DSpark): the draft step and the target's chain verify replay CUDA graphs
        self.chain = None
        if (self.draft is not None and hasattr(self.draft, "graph_chain") and tp == 1 and streams == 1
                and os.environ.get("TF_DRAFT_GRAPHS", "1") != "0"):
            from .chain_graphs import ChainGraphs

            self.chain = ChainGraphs(self.w, self.context_window + 32)
        # one stream with a tree drafter replays CUDA graphs (TF_TREE_GRAPHS=0: eager); the flag is in the rank check
        self.trees = None
        # every rank verifies a drafting request alike: one that drafts nothing itself takes the graphs without taps
        drafting = self.draft is not None or (tp > 1 and draft_dir is not None)
        if drafting and not hasattr(self.draft, "graph_chain") and streams == 1 and trees_on:
            from .tree_graphs import TreeGraphs

            self.trees = TreeGraphs(self.w, tp=tp > 1, taps=self.draft is not None, full_logits=True,
                                    log=(lambda line: print(line, flush=True)) if rank == 0 else None)
            if hasattr(self.draft, "graph_block"):
                self.draft.graphs_on = True     # the draft step too: proposals only, so outputs stay exact
        del full
        if vision and rank == 0:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, self.w.norm.device, allow_urls=vision_urls,
                                     offload=vision_offload)
        torch.cuda.empty_cache()
        from tensorfold.cuda.markers import resume_points
        from tensorfold.cuda.streams import PrefixCache

        self.eos = tuple(self.w.config.eos)
        self.model_dir = Path(model_dir)
        self.points = resume_points(model_dir)              # message starts a prefill keeps states at
        self.cache = PrefixCache(KEEP_ONE)                  # (committed ids, state, drafter snapshot)
        if tp == 1 and not many:              # one GPU: kept buffers stay inside the bytes admission left the caches
            from tensorfold.cuda.streams import KVRoom

            plan = self.capacity_plan
            spare = plan["budget_bytes"] - plan["weight_bytes_estimate"] - plan["cache_workspace_bytes_estimate"]
            self.room = KVRoom(self.cache, spare + live_kv(config(model_dir), 1, self.context_window, bits))
        # ``streams`` > 1: up to that many requests decoded together, their windows verified in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.multi = MultiDecoder(self.w, self.draft, allow_copy=allow_copy, rank=rank, world=tp,
                                      context=self.capacity_plan["cache_slots"], keep=keep, points=self.points,
                                      vision=self.vision)
            self.multi.model_dir = self.model_dir             # rank 1 compiles a request's grammar from it
            if (self.draft is not None and hasattr(self.draft, "graph_chain") and tp == 1
                    and os.environ.get("TF_DRAFT_GRAPHS", "1") != "0"):
                from .chain_graphs import ChainGraphs

                self.multi.chain = ChainGraphs(self.w, self.context_window + 32)     # a lone stream's rounds
            if (trees_on and drafting and not hasattr(self.draft, "graph_chain")
                    and (self.multi.drafts or tp > 1)):
                from .tree_graphs import MultiGraphs, TreeGraphs

                # a lone drafting stream's rounds replay CUDA graphs; several streams' too on two ranks (MultiGraphs:
                # each window padded to the same rows) and on sm_70 ranks (RoundGraphs: rows padded once, by width)
                kind = MultiGraphs if tp > 1 and not volta() else TreeGraphs
                log = (lambda line: print(line, flush=True)) if rank == 0 else None
                self.multi.trees = kind(self.w, tp=tp > 1, taps=self.multi.drafts,
                                        full_logits=self.multi.split or rank == 0, log=log)
                if rounds_on:
                    from .forward import State
                    from .round_graphs import RoundGraphs

                    together = RoundGraphs(self.w, streams, self.capacity_plan["cache_slots"] + max_rows + 2,
                                           tp=True, taps=self.multi.drafts, full_logits=self.multi.split or rank == 0,
                                           log=log)
                    st = State(self.w)
                    together.capture_all(st)
                    if round_graphs != "force":       # a width whose replay is not faster stays eager (not kept)
                        together.calibrate(st, rank, tp)
                        together = together.trimmed(st)
                    self.multi.together = together
                    del st
                if hasattr(self.draft, "graph_block"):
                    self.draft.graphs_on = True
            self.multi.calibrate(streams)
            gate = self.multi.memory_gate
            room = self.multi._room_gather() if tp > 1 else (gate.live() if gate is not None else 0)
            window = self.multi.row_bytes * self.context_window
            if rank == 0:
                print(f"[tensorfold] up to {streams} streams, each growing to {self.context_window} prompt/reply "
                      f"tokens while memory lasts, {keep} prompt states kept"
                      + (f" ({room / 2**30:.1f} GiB free for their caches on the fullest rank, "
                         f"{gate.reserve / 2**30:.2f} GiB of it kept back, {window / 2**30:.2f} GiB for one at the "
                         f"full window: {max(0, room - gate.reserve) // max(1, window)} whole windows)"
                         if gate is not None else ""), flush=True)
                curve = ", ".join(f"{r}: {ms:.1f}" for r, ms in self.multi.costs)
                print(f"[tensorfold] verify ms by rows (tree widths follow it): {curve}", flush=True)
                self.scheduler = Scheduler(self.multi, max_streams=streams)

    def _resume(self, prompt: list[int]):
        best = self.cache.longest(prompt)
        if best is not None:
            self._drop_extensions(best[0])
        return best

    def _drop_extensions(self, ids: list[int]) -> None:
        """Drop cached extensions before resuming a shorter prefix because cloned states share KV buffers and resumed writes overwrite longer prefixes."""

        n = len(ids)
        self.cache.entries = [c for c in self.cache.entries if len(c[0]) <= n or c[0][:n] != ids]

    def _remember(self, ids: list[int], st, snap) -> None:
        self.cache.add(ids, st, snap)

    def _stops(self, prompt: list[int], hit, draft: bool) -> tuple[list[int], Callable | None]:
        """Message starts past the resumed prefix, and the callback that keeps their states."""

        from tensorfold.cuda.markers import MIN_GAP

        if not draft or self.points is None:
            return [], None
        base = len(hit[0]) if hit else 0
        stops = [p for p in self.points(prompt) if p >= base + MIN_GAP]
        return stops, lambda p, st, snap: self._remember(list(prompt[:p]), st, snap)

    def _ends(self, prompt: list[int], stops: list[int]) -> bool:
        """Whether to keep the prompt end too: not when a kept message start sits just before it."""

        from tensorfold.cuda.markers import MIN_GAP

        return not (stops and len(prompt) - stops[-1] < MIN_GAP)

    def close(self) -> None:
        """Stop the concurrent scheduler's worker, so the engine's GPU memory can go (tests start several engines)."""

        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None, background=False):
        """``draft=False``: serial re-runs, no drafts; ``background``: last under ``--parallel``, yielding lanes."""

        from .decode import draft_decode, prefill

        if vision is not None and self.vision is None:
            raise ValueError("image inputs require starting this engine with --vision")
        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token safe capacity; "
                             "shorten the prompt or reserve fewer reply tokens")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        grammar = {} if constraint is None else {"constraint": constraint}     # a plain request calls as before
        if self.scheduler is not None:
            grammar.update({"background": True} if background else {})
            if vision is None:
                return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                             **grammar)
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                         vision=vision, **grammar)
        t0 = time.perf_counter()
        hit = self._resume(prompt) if draft and vision is None else None
        encoded = self.vision.encode(vision, prompt) if vision is not None else None
        if self.tp > 1:
            return self._generate_tp(prompt, max_tokens, sampling, on_tokens, hit, t0, draft, stop_eos, vision=encoded,
                                     constraint=constraint)
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
        end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
        st, pending, *kept = prefill(self.w, prompt, sampling, drafter, state=hit[1] if hit else None, room=self.room,
                                     limit=self.context_window, stops=stops, keep=keep, keep_at=end, vision=encoded,
                                     **grammar)
        if end is not None:
            self._remember(list(prompt[:end]), *kept[0])
        prefill_s = time.perf_counter() - t0
        if on_tokens([pending]):
            return {"prefill_s": prefill_s, "cached": hit[1].pos if hit else 0}
        # the cache holds the state before the last prompt token, not ``st``: the decode may commit into it
        chain = getattr(self, "chain", None)
        trees = getattr(self, "trees", None)
        if chain is not None and drafter is not None and constraint is None and vision is None:
            from .chain_graphs import chain_decode

            result = chain_decode(self.w, st, prompt, pending, max_tokens, sampling, drafter, chain,
                                  allow_copy=self.allow_copy, stop_eos=stop_eos, on_tokens=on_tokens)
        elif trees is not None and drafter is not None and constraint is None and vision is None:
            from .tree_graphs import tree_decode

            result = tree_decode(self.w, st, prompt, pending, max_tokens, sampling, drafter, trees,
                                 max_rows=self.max_rows, tree_rows=self.tree_rows, allow_copy=self.allow_copy,
                                 stop_eos=stop_eos, on_tokens=on_tokens)
        else:
            result = draft_decode(self.w, st, prompt, pending, max_tokens, sampling, drafter,
                                  max_rows=self.max_rows, tree_rows=self.tree_rows,
                                  allow_copy=self.allow_copy and draft, stop_eos=stop_eos,
                                  on_tokens=on_tokens, inplace=True, **grammar)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds,
                "cached": hit[1].pos if hit else 0, "drafts": draft, "min_rows": min(result.widths, default=0),
                "drafted": result.drafted_rows, "accepted": result.accepted_drafts}

    # several ranks: rank 0 broadcasts each request's header and prompt, every rank runs the same calls
    def _generate_tp(self, prompt, max_tokens, sampling, on_tokens, hit, t0, draft, stop_eos=True, vision=None,
                     constraint=None):
        from .decode_tp import _share, decode_tp, pack_sampling, prefill_tp
        from tensorfold.engine.grammar import pack
        from tensorfold.vision.qwen_cuda import broadcast_encoded

        dev = self.w.norm.device
        cached = hit[1].pos if hit else 0
        # the request's grammar rides after the header's fields (rank 1 compiles the same): a plain header is unchanged
        _share([1, max_tokens, cached, int(draft), *pack_sampling(sampling), int(vision is not None),
                *pack(constraint)], 0, dev)
        _share(prompt, 0, dev)
        if vision is not None:
            vision = broadcast_encoded(vision, 0, dev, hidden=self.w.config.hidden, prompt_length=len(prompt))
        grammar = {} if constraint is None else {"constraint": constraint}
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
        end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
        st, pending, *kept = prefill_tp(self.w, prompt, sampling, 0, drafter, state=hit[1] if hit else None,
                                        limit=self.context_window, stops=stops, keep=keep, keep_at=end,
                                        vision=vision, **grammar)
        if end is not None:
            self._remember(list(prompt[:end]), *kept[0])
        prefill_s = time.perf_counter() - t0
        stop_now = bool(on_tokens([pending]))
        # rank 0 alone decides where a reply ends; the others follow its windows (no header field needed)
        result = decode_tp(self.w, st, prompt, pending, 1 if stop_now else max_tokens, sampling, 0, drafter,
                           max_rows=self.max_rows, allow_copy=self.allow_copy and draft, stop_eos=stop_eos,
                           on_tokens=on_tokens, inplace=True,
                           runner=self._tree_runner(drafter is not None, constraint, vision),
                           **grammar)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds, "cached": cached,
                "drafts": draft, "min_rows": min(result.widths, default=0), "drafted": result.drafted_rows,
                "accepted": result.accepted_drafts}

    def _tree_runner(self, drafts: bool, constraint, vision):
        """The tree graphs for a request on several ranks; every rank decides alike from shared fields."""

        trees = getattr(self, "trees", None)
        return trees if trees is not None and drafts and constraint is None and vision is None else None

    def follow(self) -> None:
        """Ranks past 0: mirror every request rank 0 serves, forever."""

        if self.multi is not None:
            self.multi.follow()
            return

        from .decode_tp import SAMPLING_WORDS as W, _share, decode_tp, prefill_tp, unpack_sampling
        from tensorfold.vision.qwen_cuda import broadcast_encoded

        dev = self.w.norm.device
        while True:
            header = _share(None, self.rank, dev)
            _, max_tokens, cached, draft = header[:4]
            sampling = unpack_sampling(header[4:4 + W])
            prompt = _share(None, self.rank, dev)
            vision = (broadcast_encoded(None, self.rank, dev, hidden=self.w.config.hidden, prompt_length=len(prompt))
                      if len(header) > 4 + W and header[4 + W] else None)
            packed = header[5 + W:]
            grammar = {}
            if packed:                                  # compiled here as on rank 0
                from tensorfold.engine import grammar as g

                grammar = {"constraint": g.compiler(self, self.model_dir, self.eos).follow(packed)}
            drafter = self.draft if draft else None
            hit = self.cache.named(prompt, cached) if cached else None
            if cached and hit is None:
                raise RuntimeError(f"rank {self.rank} has no cached state for the {cached} tokens rank 0 resumes "
                                   "from")
            if hit is not None:
                self._drop_extensions(hit[0])
            if drafter is not None:             # a two-rank drafter: mirror rank 0's drafter state
                drafter.restore(hit[2] if hit is not None else
                                ([None] * drafter.layers, [None] * drafter.layers, 0, 0))
            stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
            # the same entries as rank 0
            end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
            st, pending, *kept = prefill_tp(self.w, prompt, sampling, self.rank, drafter, state=hit[1] if hit else None,
                                            limit=self.context_window, stops=stops, keep=keep, keep_at=end,
                                            vision=vision, **grammar)
            if end is not None:
                self._remember(list(prompt[:end]), *kept[0])
            result = decode_tp(self.w, st, prompt, pending, max_tokens, sampling, self.rank, drafter,
                               max_rows=self.max_rows, inplace=True,
                               runner=self._tree_runner(bool(draft), grammar.get("constraint"), vision), **grammar)
