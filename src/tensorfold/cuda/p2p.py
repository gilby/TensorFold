"""Ranks on one machine (two to eight): row-parallel sums in one launch, not a collective.

A rank stages its partial in a buffer its peers have mapped, and every rank adds all the partials in rank order in
fp32, rounding once to bf16: the bits of ``gather_rank_partials``, at a kernel launch's host cost instead of an NCCL
call's.
- ``PeerSum``: the GPUs map each other (NVLink / PCIe P2P); the staging buffers are device memory.
- ``HostSum``: they cannot (PCIe GPUs in a VM); the staging buffers and flags are one shared page-locked host buffer
  every process maps, so the same kernel writes and reads the partials over PCIe.
Each block's round counter lives in device memory, so a sum captured in a CUDA graph replays with the next round.
``TF_TP_P2P=0`` turns both off (NCCL), ``TF_TP_HOST_SUM=0`` only the host one; sums larger than a stage keep NCCL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

STAGE_BYTES = 8 << 20       # a rank's staging buffer: 409 fp32 rows of 5120 (decode windows; prompts above use NCCL)
SUM_BYTES = 1 << 20         # partials up to this size are summed by one kernel; larger ones are gathered by copies


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import VOLTA, load

    here = Path(__file__).parent
    return load(name="tensorfold_p2p_v6", sources=[str(here / "p2p.cu")], need=VOLTA, extra_cuda_cflags=["-O3"],
                verbose=False)


class _Group:
    """torch.distributed's group, or an engine's communicator (``all_gather(send, recv)``, ``barrier()``, ``world``)."""

    def __init__(self, group=None, comm=None) -> None:
        self.group, self.comm = group, comm

    @property
    def world(self) -> int:
        if self.comm is not None:
            return int(self.comm.world)
        import torch.distributed as dist

        return dist.get_world_size(self.group)

    def gather(self, mine: torch.Tensor) -> torch.Tensor:
        """Every rank's ``mine`` (a 1-D CUDA tensor), rank order: [world, n]."""

        every = torch.empty((self.world, mine.numel()), dtype=mine.dtype, device=mine.device)
        if self.comm is not None:
            self.comm.all_gather(mine.contiguous(), every.view(-1))
            torch.cuda.current_stream().synchronize()
        else:
            import torch.distributed as dist

            dist.all_gather_into_tensor(every, mine.contiguous(), group=self.group)
        return every

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.barrier()
        else:
            import torch.distributed as dist

            dist.barrier(group=self.group)


def _world(group) -> int:
    return group.world if isinstance(group, _Group) else _Group(group).world


class PeerSum:
    """This rank's staging and flags, and every peer's, mapped once at startup."""

    kind = "P2P (the peer GPUs mapped)"
    backoff = 0                 # ns a waiting block sleeps between flag polls (0: spin)

    def __init__(self, rank: int, group=None, stage_bytes: int = STAGE_BYTES) -> None:
        g = group if isinstance(group, _Group) else _Group(group)
        ext = _ext()
        world = g.world
        self.rank, self.world, self.stage_bytes = rank, world, int(stage_bytes)
        self.half = -(-self.stage_bytes // 16) * 16              # a round's buffer: two of them, by round parity
        stage, sig = ext.alloc(2 * self.half), ext.alloc(ext.signal_bytes())
        mine = torch.cat([ext.handle(stage), ext.handle(sig)]).cuda()
        every = g.gather(mine).cpu()
        half = mine.numel() // 2
        stages, sigs = [], []
        for q in range(world):
            if q == rank:
                stages.append(stage)
                sigs.append(sig)
            else:
                stages.append(ext.open(every[q, :half].contiguous()))
                sigs.append(ext.open(every[q, half:].contiguous()))
        self._addresses(stages, sigs)
        torch.cuda.synchronize()
        g.barrier()

    gather = None               # copies through host memory (``HostSum``); a mapped peer sums in one kernel instead
    captured: int | None = None     # sums recorded by the capture in progress

    def _addresses(self, stages: list[int], sigs: list[int]) -> None:
        self.stages = torch.tensor(stages, dtype=torch.int64)
        self.sigs = torch.tensor(sigs, dtype=torch.int64)
        self.rounds = torch.zeros(_ext().blocks(), dtype=torch.int64, device="cuda")
        self.meets = torch.zeros(2, dtype=torch.int64, device="cuda")

    def begin_capture(self) -> None:
        """Count the sums a capture records."""

        self.captured = 0

    def end_capture(self) -> int:
        """End a capture; return the sums it holds."""

        calls, self.captured = self.captured or 0, None
        return calls

    def before_replay(self, calls: int) -> None:
        """Nothing to set: a replayed sum reads and advances its block's round on the device."""

    def small(self, local: torch.Tensor) -> bool:
        """Whether one sum kernel takes this partial (decode rows); a larger one is gathered where ``gather`` is set."""

        return local.numel() * local.element_size() <= self.half

    def fits(self, local: torch.Tensor) -> bool:
        return local.numel() * local.element_size() <= self.stage_bytes

    def __call__(self, local: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        local = local.contiguous()
        if local.data_ptr() % 16:
            local = local.clone()
        out = torch.empty(local.shape, dtype=torch.bfloat16, device=local.device)
        if self.captured is not None:
            self.captured += 1
        _ext().p2p_sum(local, self.stages, self.sigs, out, self.rank, self.rounds, self.backoff, self.half)
        return out if dtype == torch.bfloat16 else out.to(dtype)


NAME_BYTES = 128


class HostSum(PeerSum):
    """Every rank's staging and flags in one ``/dev/shm`` buffer, page-locked and mapped by each GPU: [the ranks'
    stages, the ranks' flags]. Rank 0 creates it (zeroed), the others open it, rank 0 unlinks the name once all hold
    the mapping."""

    kind = "host memory (the GPUs cannot map each other: the partials go over PCIe through shared page-locked memory)"

    backoff = int(os.environ.get("TF_TP_HOST_SPIN_NS", "500"))   # flags polled over PCIe: gently

    def __init__(self, rank: int, group=None, stage_bytes: int = STAGE_BYTES) -> None:
        import mmap
        import uuid

        g = group if isinstance(group, _Group) else _Group(group)
        ext = _ext()
        world = g.world
        self.rank, self.world, self.stage_bytes = rank, world, int(stage_bytes)
        self.half = min(SUM_BYTES, -(-self.stage_bytes // 16) * 16)     # the sum kernel's two buffers by round parity
        region = 2 * self.half + -(-self.stage_bytes // 16) * 16        # then the copies' stage
        sig = ext.signal_bytes()
        size = -(-(world * (region + sig)) // mmap.PAGESIZE) * mmap.PAGESIZE
        name = torch.zeros(NAME_BYTES, dtype=torch.uint8)
        if rank == 0:
            path = f"/dev/shm/tensorfold-sum-{os.getpid()}-{uuid.uuid4().hex[:12]}"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)                                  # zero-filled: every flag starts below round 1
            raw = path.encode()
            name[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
        every = g.gather(name.cuda())
        if rank != 0:
            raw = bytes(b for b in every[0].cpu().tolist() if b)
            path = raw.decode()
            fd = os.open(path, os.O_RDWR)
        try:
            self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        import ctypes

        host = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        base = ext.host_map(host, size)                             # this process's device address of the buffer
        self._addresses([base + q * region for q in range(world)],
                        [base + world * region + q * sig for q in range(world)])
        self.host = [torch.frombuffer(self._map, dtype=torch.uint8, count=self.stage_bytes,
                                      offset=q * region + 2 * self.half)
                     for q in range(world)]          # registered (page-locked): copies to and from them are DMA
        torch.cuda.synchronize()
        g.barrier()
        if rank == 0:
            os.unlink(path)                                         # all mapped: the name is not needed any more

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        """Every rank's ``local`` in rank order, [world, *shape]: each copies its partial into its stage, all meet,
        each copies the others' stages back, all meet again (the next round may overwrite them). Stream-ordered."""

        local = local.contiguous()
        n = local.numel() * local.element_size()
        out = torch.empty((self.world, *local.shape), dtype=local.dtype, device=local.device)
        view = lambda q: self.host[q][:n].view(local.dtype).view(local.shape)      # noqa: E731
        view(self.rank).copy_(local, non_blocking=True)
        _ext().meet(self.sigs, self.rank, 0, self.meets, self.backoff)
        for q in range(self.world):
            out[q].copy_(local if q == self.rank else view(q), non_blocking=True)
        _ext().meet(self.sigs, self.rank, 1, self.meets, self.backoff)
        return out

    def shares(self, rows: int) -> list[int]:
        """Row bounds of each rank's share of a reduce: by ``TF_TP_SUM_SHARES`` weights (a rank with a slower host
        link may take fewer rows, or none), else even. Any shares give the same bits."""

        raw = os.environ.get("TF_TP_SUM_SHARES", "").strip()
        weights = [int(x) for x in raw.split(",")] if raw else [1] * self.world
        if len(weights) != self.world or min(weights) < 0 or not sum(weights):
            raise ValueError(f"TF_TP_SUM_SHARES: {self.world} non-negative whole weights, not {raw!r}")
        total, cum, out = sum(weights), 0, [0]
        for x in weights:
            cum += x
            out.append(rows * cum // total)
        return out

    def reduce(self, local: torch.Tensor) -> torch.Tensor:
        """``gather``'s rank-ordered fp32 sum rounded once to ``local``'s dtype, the same bits, through a reduce-scatter
        and an all-gather: each rank copies the rows other ranks reduce into its stage, adds its own share of the rows
        from every stage in rank order, posts that share in its own stage (no other rank reads those rows of it), and
        copies the others' shares back."""

        local = local.contiguous()
        rows, world = local.shape[0], self.world
        n = local.numel() * local.element_size()
        view = lambda q: self.host[q][:n].view(local.dtype).view(local.shape)      # noqa: E731
        bounds = self.shares(rows)
        a, b = bounds[self.rank], bounds[self.rank + 1]
        mine_stage = view(self.rank)
        if a:
            mine_stage[:a].copy_(local[:a], non_blocking=True)
        if b < rows:
            mine_stage[b:].copy_(local[b:], non_blocking=True)
        _ext().meet(self.sigs, self.rank, 0, self.meets, self.backoff)
        out = torch.empty_like(local)
        if b > a:
            mine = torch.empty((world, b - a, *local.shape[1:]), dtype=local.dtype, device=local.device)
            for q in range(world):
                mine[q].copy_(local[a:b] if q == self.rank else view(q)[a:b], non_blocking=True)
            acc = mine[0].float()
            for q in range(1, world):
                acc = acc + mine[q].float()
            out[a:b] = acc.to(local.dtype)
            mine_stage[a:b].copy_(out[a:b], non_blocking=True)
        _ext().meet(self.sigs, self.rank, 1, self.meets, self.backoff)
        for q in range(world):
            if q != self.rank and bounds[q + 1] > bounds[q]:
                out[bounds[q]:bounds[q + 1]].copy_(view(q)[bounds[q]:bounds[q + 1]], non_blocking=True)
        _ext().meet(self.sigs, self.rank, 0, self.meets, self.backoff)   # no stage is rewritten before all have read
        return out


_PEER: PeerSum | None = None


def _same_machine(group=None) -> bool:
    """Whether every rank runs on this machine (one boot): only then can they share host memory."""

    import hashlib
    import socket

    g = group if isinstance(group, _Group) else _Group(group)
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return False
    digest = hashlib.sha256(f"{socket.gethostname()}/{boot}".encode()).digest()[:8]
    every = g.gather(torch.tensor(list(digest), dtype=torch.uint8, device="cuda"))
    return bool((every == every[0]).all())


def install(rank: int, group=None, *, comm=None, stage_bytes: int = STAGE_BYTES) -> bool:
    """Map the peers for row-parallel sums when every GPU can reach every other, else share host memory when all ranks
    are on this machine (every rank calls this together); False: NCCL sums. ``comm``: an engine's communicator in
    place of torch.distributed; ``stage_bytes``: the largest partial a rank stages."""

    global _PEER
    g = _Group(group, comm)
    me = torch.cuda.current_device()
    others = [d for d in range(torch.cuda.device_count()) if d != me]
    on = os.environ.get("TF_TP_P2P", "1") != "0"
    reach = [torch.cuda.can_device_access_peer(me, d) for d in others]
    able = int(on and bool(reach) and (any(reach) if g.world == 2 else all(reach)))
    flags = torch.tensor([able, int(on and os.environ.get("TF_TP_HOST_SUM", "1") != "0")], dtype=torch.int32,
                         device="cuda")
    every = g.gather(flags)
    if bool(every[:, 0].all()):
        _PEER = PeerSum(rank, g, stage_bytes)
        return True
    if bool(every[:, 1].all()) and _same_machine(g):
        _PEER = HostSum(rank, g, stage_bytes)
        return True
    return False


def peer() -> PeerSum | None:
    return _PEER


_SECOND: list = []


def second() -> PeerSum | None:
    """A second host-memory sum (its own stages, flags and rounds) for sums on a second stream beside the first's,
    made on first use by every rank together (as ``install``); None where the first is not a ``HostSum``."""

    if not isinstance(_PEER, HostSum):
        return None
    if not _SECOND:
        _SECOND.append(HostSum(_PEER.rank, stage_bytes=_PEER.stage_bytes))
    return _SECOND[0]
