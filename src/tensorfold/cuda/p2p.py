"""Two ranks on one machine: row-parallel sums in one launch, not a collective.

A rank stages its partial in a buffer the peer has mapped, and each rank adds rank 0's partial and rank 1's in fp32,
rounding once to bf16: the bits of ``gather_rank_partials`` (two fp32 values add the same in either order), at a
kernel launch's host cost instead of an NCCL call's (on sm1's VM, ~10 us against ~250 us with the add and the cast).
- ``PeerSum``: both GPUs map each other (NVLink / PCIe P2P); the staging buffers are device memory.
- ``HostSum``: they cannot (PCIe GPUs in a VM, e.g. the V100s on Volta): the staging buffers and flags are one shared
  page-locked host buffer both processes map, so the same kernel writes and reads the partials over PCIe.
``TF_TP_P2P=0`` turns both off (NCCL), ``TF_TP_HOST_SUM=0`` only the host one; sums larger than a stage keep NCCL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

STAGE_BYTES = 8 << 20       # a rank's staging buffer: 409 fp32 rows of 5120 (decode windows; prompts above use NCCL)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import VOLTA, load

    here = Path(__file__).parent
    return load(name="tensorfold_p2p_v4", sources=[str(here / "p2p.cu")], need=VOLTA, extra_cuda_cflags=["-O3"],
                verbose=False)


class PeerSum:
    """This rank's staging and flags, and the peer's, mapped once at startup."""

    kind = "P2P (the peer GPU mapped)"
    backoff = 0                 # ns a waiting block sleeps between flag polls (0: spin)

    def __init__(self, rank: int, group=None) -> None:
        import torch.distributed as dist

        ext = _ext()
        self.rank, self.round = rank, 0
        self.captured: int | None = None        # calls recorded by the capture in progress
        self.base = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.stage = ext.alloc(STAGE_BYTES)
        self.sig = ext.alloc(ext.signal_bytes())
        mine = torch.cat([ext.handle(self.stage), ext.handle(self.sig)]).cuda()
        both = torch.empty((2, mine.numel()), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(both, mine, group=group)
        peer = both[1 - rank].cpu()
        half = peer.numel() // 2
        self.peer_stage, self.peer_sig = ext.open(peer[:half].contiguous()), ext.open(peer[half:].contiguous())
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def fits(self, local: torch.Tensor) -> bool:
        return local.numel() * local.element_size() <= STAGE_BYTES

    def __call__(self, local: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        local = local.contiguous()
        out = torch.empty(local.shape, dtype=torch.bfloat16, device=local.device)
        if self.captured is not None:          # captured: round = device base + this call's offset
            self.captured += 1
            _ext().p2p_sum(local, self.stage, self.peer_stage, self.sig, self.peer_sig, out, self.rank, self.captured,
                           self.base.data_ptr(), self.backoff)
        else:
            self.round += 1
            _ext().p2p_sum(local, self.stage, self.peer_stage, self.sig, self.peer_sig, out, self.rank, self.round, 0,
                           self.backoff)
        return out if dtype == torch.bfloat16 else out.to(dtype)

    def begin_capture(self) -> None:
        self.captured = 0

    def end_capture(self) -> int:
        """End a capture; return the sums it holds."""

        calls, self.captured = self.captured or 0, None
        return calls

    def before_replay(self, calls: int) -> None:
        """Set the device base so a replay of ``calls`` sums takes the next rounds."""

        self.base.fill_(self.round)             # stream-ordered before the replay
        self.round += calls


NAME_BYTES = 128


class HostSum(PeerSum):
    """Both ranks' staging and flags in one ``/dev/shm`` buffer, page-locked and mapped by each GPU: [rank 0 stage,
    rank 1 stage, rank 0 flags, rank 1 flags]. Rank 0 creates it (zeroed), rank 1 opens it, rank 0 unlinks the name
    once both hold the mapping."""

    kind = "host memory (the GPUs cannot map each other: both partials over PCIe through shared page-locked memory)"

    backoff = int(os.environ.get("TF_TP_HOST_SPIN_NS", "500"))   # flags polled over PCIe: gently

    def __init__(self, rank: int, group=None) -> None:
        import mmap
        import uuid

        import torch.distributed as dist

        ext = _ext()
        sig = ext.signal_bytes()
        size = -(-(2 * STAGE_BYTES + 2 * sig) // mmap.PAGESIZE) * mmap.PAGESIZE
        self.rank, self.round = rank, 0
        self.captured: int | None = None
        self.base = torch.zeros(1, dtype=torch.int64, device="cuda")
        name = torch.zeros(NAME_BYTES, dtype=torch.uint8)
        if rank == 0:
            path = f"/dev/shm/tensorfold-sum-{os.getpid()}-{uuid.uuid4().hex[:12]}"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)                                  # zero-filled: every flag starts below round 1
            raw = path.encode()
            name[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
        both = torch.empty((2, NAME_BYTES), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(both, name.cuda(), group=group)
        if rank != 0:
            raw = bytes(b for b in both[0].cpu().tolist() if b)
            path = raw.decode()
            fd = os.open(path, os.O_RDWR)
        try:
            self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        import ctypes

        host = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        base = ext.host_map(host, size)                             # this process's device address of the buffer
        stages = [base, base + STAGE_BYTES]
        sigs = [base + 2 * STAGE_BYTES, base + 2 * STAGE_BYTES + sig]
        self.stage, self.peer_stage = stages[rank], stages[1 - rank]
        self.sig, self.peer_sig = sigs[rank], sigs[1 - rank]
        torch.cuda.synchronize()
        dist.barrier(group=group)
        if rank == 0:
            os.unlink(path)                                         # both mapped: the name is not needed any more


_PEER: PeerSum | None = None


def _same_machine(group=None) -> bool:
    """Whether both ranks run on this machine (one boot): only then can they share host memory."""

    import hashlib
    import socket

    import torch.distributed as dist

    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return False
    digest = hashlib.sha256(f"{socket.gethostname()}/{boot}".encode()).digest()[:8]
    mine = torch.tensor(list(digest), dtype=torch.uint8, device="cuda")
    both = torch.empty((2, 8), dtype=torch.uint8, device="cuda")
    dist.all_gather_into_tensor(both, mine, group=group)
    return bool(torch.equal(both[0], both[1]))


def install(rank: int, group=None) -> bool:
    """Map the peer for row-parallel sums when both GPUs can reach each other, else share host memory when both ranks
    are on this machine (both ranks call this together); False: NCCL sums."""

    global _PEER
    import torch.distributed as dist

    me = torch.cuda.current_device()
    others = [d for d in range(torch.cuda.device_count()) if d != me]
    on = os.environ.get("TF_TP_P2P", "1") != "0"
    able = int(on and any(torch.cuda.can_device_access_peer(me, d) for d in others))
    flags = torch.tensor([able, int(on and os.environ.get("TF_TP_HOST_SUM", "1") != "0")], dtype=torch.int32,
                         device="cuda")
    both = torch.empty((2, 2), dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(both, flags, group=group)
    if bool(both[:, 0].all()):
        _PEER = PeerSum(rank, group)
        return True
    if bool(both[:, 1].all()) and _same_machine(group):
        _PEER = HostSum(rank, group)
        return True
    return False


def peer() -> PeerSum | None:
    return _PEER
