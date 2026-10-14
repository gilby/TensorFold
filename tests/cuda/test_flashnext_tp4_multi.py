"""Flash Next's concurrent decoder on four ranks (--tp 4 --parallel N), and on two through the same code.

The ranks run as threads on one GPU, their all-gathers through host memory; rank 0 drives a MultiDecoder and sends
every admission, round and completion to each follower. Checks: the link delivers each message to every
follower in order (rendezvous store and TCP); the plan check names an out-of-step rank and a readiness check holds
only when it holds on every rank; streams decoded together emit what the ranks' tensor-parallel serial decoding
emits and what each stream emits alone, greedy and sampled, drafted and not. Volta (sm_70) uses the tiny NVFP4
checkpoint, other cards the tiny MLX one.
"""

import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import Link, MultiDecoder, OutOfStep  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_tp import TwoRanks  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

WORLD = 4
VOLTA = torch.cuda.get_device_capability()[0] == 7
PROMPTS = [[5, 17, 99, 250], [100, 7, 64, 30, 11, 12], [13], [8, 8, 9, 200, 31], [40, 41, 42]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=0, top_p=0.9),
             Sampling(seed=9, top_k=50, top_p=0.95), Sampling(seed=11, temperature=0.7, top_k=20, min_p=0.1)]


class _Store:
    """The TCP store's set / wait / get / delete_key, in memory."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.cv = threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cv.notify_all()

    def wait(self, keys, timeout=None) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.data for k in keys), timeout=120):
                raise TimeoutError(keys)

    def get(self, key) -> bytes:
        return self.data[key]

    def delete_key(self, key) -> None:
        with self.cv:
            self.data.pop(key, None)


class _Hub:
    def __init__(self, world: int) -> None:
        self.world, self.slots = world, [None] * world
        self.barrier = threading.Barrier(world, timeout=600)


class _ThreadComm:
    """``comm.NCCL``'s all_gather and barrier for ranks that are threads of one process on one GPU."""

    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = send
        self.hub.barrier.wait()
        n, flat = send.numel(), recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()

    def barrier(self) -> None:
        self.hub.barrier.wait()


def _run_ranks(fn, items: list) -> list:
    """fn(rank, item) on every rank at once (threads, each on its own CUDA stream); results in rank order."""

    hub = _Hub(len(items))
    for r, item in enumerate(items):
        item.w.comm = _ThreadComm(hub, r)
    results: list = [None] * len(items)
    errors: list = []
    streams = [torch.cuda.Stream() for _ in items]

    def body(r: int) -> None:
        try:
            with torch.no_grad(), torch.cuda.stream(streams[r]):
                results[r] = fn(r, items[r])
                torch.cuda.current_stream().synchronize()
        except BaseException as exc:        # noqa: BLE001  (reported below; unblock the other ranks)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(items))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


def test_the_store_link_gives_every_follower_each_message_in_order():
    store = _Store()
    leader = Link(store, world=WORLD)
    followers = [Link(store, rank=r, world=WORLD) for r in range(1, WORLD)]
    sent = [["admit", 0, [1, 2]], ["round", [], {"x": 1}], ["finish", [0]], ["stop"]]
    for op in sent:
        leader.send(op)
    for f in followers:
        assert [f.receive() for _ in sent] == sent
    assert not store.data                                  # each follower took and deleted its own copies


def test_the_tcp_link_gives_every_follower_each_message_in_order():
    store = _Store()
    host = "127.0.0.1"
    leader = Link(store, rank=0, host=host, world=WORLD)
    sent = [["admit", 1, [9] * 5000], ["round", [3], {"passes": []}], ["stop"]]
    got: dict[int, list] = {}

    def follow(r: int) -> None:
        link = Link(store, rank=r, host=host, world=WORLD)
        got[r] = [link.receive() for _ in sent]
        assert link.receive() is None                    # the leader closed: the follower stops

    threads = [threading.Thread(target=follow, args=(r,)) for r in range(1, WORLD)]
    for th in threads:
        th.start()
    for op in sent:
        leader.send(op)
    for sock in leader.socks:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
    for th in threads:
        th.join(timeout=60)
    assert got == {r: sent for r in range(1, WORLD)}


def _agree_on_ranks(plans: list, checks: list):
    """``TwoRanks._agree`` on every rank at once: the plan and checks each rank proposes; results or errors."""

    ranks = [SimpleNamespace(w=SimpleNamespace(comm=None)) for _ in plans]

    def body(r, me):
        try:
            return TwoRanks._agree(me, "round", plans[r], checks[r])
        except OutOfStep as exc:
            return exc

    return _run_ranks(body, ranks)


def test_the_plan_check_names_the_rank_out_of_step_and_every_rank_refuses():
    got = _agree_on_ranks([[1, 2], [1, 2], [1, 3], [1, 2]], [(True,)] * WORLD)
    assert all(isinstance(g, OutOfStep) and "rank 2" in str(g) for g in got)


def test_a_check_holds_only_when_it_holds_on_every_rank():
    """A cache growth that fits ranks 2 and 3 but not the fuller ranks 0 and 1 (more experts there) fits nowhere."""

    got = _agree_on_ranks([[7]] * WORLD, [(True, False), (True, False), (True, True), (True, True)])
    assert got == [(True, False)] * WORLD
    got = _agree_on_ranks([[7]] * WORLD, [(True, True), (True, True), (True, True), (False, True)])
    assert got == [(False, True)] * WORLD


@pytest.fixture(scope="module", params=[4, 2], ids=["tp4", "tp2"])
def ranks(request, tmp_path_factory):
    world = request.param
    if world == 4 and not VOLTA:
        pytest.skip("four ranks serve NVFP4 checkpoints on sm_70; the tiny MLX checkpoint splits over two")
    root = tmp_path_factory.mktemp(f"flashnext_tp{world}_multi")
    if VOLTA:
        from nvfp4_tiny import write

        write(root, hidden=512, heads=4, kv_heads=2, hd=64, nk=16, nv=48, moe_width=320, shared_width=320,
              experts=4, ple_bf16=True)
        return [load(root, tp=(r, world), draft_vocab=128) for r in range(world)]
    from test_flashnext_tp import _checkpoint

    _checkpoint(root)
    return [load(root, tp=(r, world)) for r in range(world)]


def _serial(ranks, prompt, sampling, count, kv_dtype):
    """The ranks' tensor-parallel serial decoding (every rank emits the same tokens)."""

    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype) for w in ranks]
    got = _run_ranks(lambda r, e: serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens, engines)
    assert all(g == got[0] for g in got[1:])
    return got[0]


def _ranks_run(ranks, kv_dtype, drive, slots=4, **options):
    """rank 0: ``drive(decoder)`` with its link to the followers, then stop; the other ranks follow. Returns drive's."""

    world = len(ranks)

    decs = [MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, graphs=False,
                         **options) for w in ranks]
    store = _Store()

    def body(r, dec):
        if r == 0:
            dec.link = Link(store, world=world)
            try:
                return drive(dec)
            finally:
                dec.link.send(["stop"])
        dec.follow(Link(store, rank=r, world=world))
        return dec

    out = _run_ranks(body, decs)
    for follower in out[1:]:
        assert not follower.streams and not follower.filling
        assert len(follower.free) + len({id(k[1]) for k in follower.kept}) == slots
    return out[0]


_SOLO: dict[tuple, list[int]] = {}


def _alone(ranks, i, count, kv_dtype):
    """Request ``i`` served by itself on these ranks with drafts (its solo run)."""

    key = (len(ranks), i, count, kv_dtype)
    if key not in _SOLO:
        def drive(dec):
            s = Stream(list(PROMPTS[i]), count, SAMPLINGS[i])
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            return s.out
        _SOLO[key] = _ranks_run(ranks, kv_dtype, drive)
    return _SOLO[key]


@pytest.mark.parametrize("users", [1, 2, 4])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_ranks_every_reply_equals_its_serial_run_and_its_solo_run(ranks, users, kv_dtype):
    """Greedy and sampled streams decoded together: each equals its serial (undrafted) run and its run alone with
    drafts."""

    count = 16

    def drive(dec):
        streams = [Stream(list(p), count, s) for p, s in zip(PROMPTS[:users], SAMPLINGS[:users])]
        for s in streams:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return [s.out for s in streams]

    together = _ranks_run(ranks, kv_dtype, drive)
    for i, got in enumerate(together):
        assert got == _serial(ranks, PROMPTS[i], SAMPLINGS[i], count, kv_dtype), (users, i)
        assert got == _alone(ranks, i, count, kv_dtype), (users, i)


def test_ranks_a_stream_joining_mid_flight_and_one_undrafted_equal_serial(ranks):
    refs = [_serial(ranks, p, s, 20, "bf16") for p, s in zip(PROMPTS, SAMPLINGS)]

    def drive(dec):
        streams = []
        for i, (prompt, sampling) in enumerate(zip(PROMPTS, SAMPLINGS)):
            got: list[int] = []
            s = Stream(prompt, 20, sampling, draft=i != 2, emit=lambda new, got=got: got.extend(new))
            if i < 4:
                dec.admit(s)
            streams.append((s, got))
        joined = False
        while dec.live():
            dec.finish(dec.round())
            if not joined:                                    # the fifth joins after the first round
                dec.admit(streams[4][0])
                joined = True
        return streams

    for i, (s, got) in enumerate(_ranks_run(ranks, "bf16", drive, slots=5)):
        assert got == refs[i] and s.out == refs[i], i


def test_ranks_a_client_leaving_on_rank_0_alone_keeps_every_rank_in_step(ranks):
    refs = [_serial(ranks, p, s, 20, "int8") for p, s in zip(PROMPTS[:3], SAMPLINGS[:3])]

    def drive(dec):
        outs = []
        for i, (prompt, sampling) in enumerate(zip(PROMPTS[:3], SAMPLINGS[:3])):
            got: list[int] = []
            emit = (lambda new, got=got: (got.extend(new), len(got) >= 4)[1]) if i == 1 else \
                (lambda new, got=got: got.extend(new))
            s = Stream(prompt, 20, sampling, emit=emit)
            dec.admit(s)
            outs.append((s, got))
        while dec.live():
            dec.finish(dec.round())
        after: list[int] = []
        s = Stream(PROMPTS[3], 20, SAMPLINGS[3], emit=lambda new: after.extend(new))
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return outs, after

    outs, after = _ranks_run(ranks, "int8", drive)
    assert outs[0][1] == refs[0] and outs[2][1] == refs[2]
    assert outs[1][1] == refs[1][:len(outs[1][1])] and 4 <= len(outs[1][1]) < 20
    assert after == _serial(ranks, PROMPTS[3], SAMPLINGS[3], 20, "int8")


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    root = tmp_path_factory.mktemp("flashnext_one_multi")
    if VOLTA:
        from nvfp4_tiny import write

        write(root, hidden=512, heads=4, kv_heads=2, hd=64, nk=16, nv=48, moe_width=320, shared_width=320,
              experts=4, ple_bf16=True)
        return load(root, draft_vocab=128)
    from test_flashnext_tp import _checkpoint

    _checkpoint(root)
    return load(root)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_one_gpu_streams_together_equal_serial_decoding(one, kv_dtype):
    """One GPU, every stream's rows in one round: each stream, drafted or not, emits its serial tokens (on sm_70 the
    rounds' attention takes the one-stream kernel, whose partials the shared Triton kernel's would not equal)."""

    count = 40
    refs = []
    for p, s in zip(PROMPTS, SAMPLINGS):
        e = Engine(one, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, p, s), count, s, stop_eos=True).tokens)
    dec = MultiDecoder(one, slots=5, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, graphs=False)
    streams = [Stream(list(p), count, s, draft=i % 2 == 0) for i, (p, s) in enumerate(zip(PROMPTS, SAMPLINGS))]
    with torch.no_grad():
        for s in streams:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
    for i, s in enumerate(streams):
        assert s.out == refs[i], i


def test_one_gpu_lone_requests_keep_the_graph_slot_and_its_graphs_when_every_slot_holds_a_keep(one):
    """Every slot holds a kept prompt end: a lone request evicts the oldest other keep rather than re-pointing the graph
    slot, so its graphs survive request after request; every reply equals serial decoding."""

    count = 24
    dec = MultiDecoder(one, slots=3, capacity=1024, depth=3, confidence=0.3)
    assert dec.solo is not None

    def run(prompts):
        streams = [Stream(list(p), count, None) for p in prompts]
        with torch.no_grad():
            for s in streams:
                dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
        return streams

    def fresh(p):
        e = Engine(one, capacity=1024, max_rows=8, prefill_rows=16)
        return serial_decode(e, prefill(e, p, None), count, None, stop_eos=True).tokens

    lone = [[i + 3, 50 + i, 9, 200 - i, 31] for i in range(6)]
    run(PROMPTS[:3])                                     # three keeps, one in every slot
    graphs, slot = None, None
    for i, p in enumerate(lone):
        (s,) = run([p])
        assert s.out == fresh(p), i
        if i >= 1:                                       # the slot reached its size with the first lone request
            assert dec.solo.st is slot and dec.solo.graphs is graphs, i
        graphs, slot = dec.solo.graphs, dec.solo.st
