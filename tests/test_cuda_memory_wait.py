"""Requests memory can't hold yet wait in order and start when a stream finishes; a stream that gives way for memory
mid-reply replays from its prompt (its sent tokens owed, not resent) instead of failing; a wait has a limit."""

import threading
import time

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.memory_gate import NoRoom  # noqa: E402
from tensorfold.cuda.scheduler import MEMORY_WAIT_ENV, Scheduler  # noqa: E402

WAIT = 10.0


class Decoder:
    """Streams emit token 100 + position; ``room`` streams fit at once; ``give_way`` sids yield once at 2 tokens."""

    def __init__(self, room: int, give_way=(), step_s: float = 0.002):
        self.room, self.give_way, self.step_s = room, set(give_way), step_s
        self.streams, self.next_id, self.admitted = {}, 0, []

    def live(self) -> int:
        return len(self.streams)

    def admit(self, s) -> None:
        if len(self.streams) >= self.room:
            raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
        s.sid, self.next_id = self.next_id, self.next_id + 1
        self.streams[s.sid] = s
        self.admitted.append(s.prompt[0])

    def round(self):
        time.sleep(self.step_s)
        done = []
        for s in list(self.streams.values()):
            if s.prompt[0] in self.give_way and len(s.out) >= 2 and len(self.streams) > 1:
                self.give_way.discard(s.prompt[0])
                s.preempted, s.done = True, True
                done.append(s)
                continue
            s.take([100 + len(s.out)])
            if len(s.out) >= s.count:
                s.done = True
                done.append(s)
        return done

    def finish(self, done) -> None:
        for s in done:
            self.streams.pop(s.sid, None)

    def drop(self):
        live = list(self.streams.values())
        self.streams.clear()
        return live


class Request:
    def __init__(self, sched, first: int, count: int):
        self.tokens, self.got = [], []
        self.thread = threading.Thread(target=self._go, args=(sched, first, count), daemon=True)
        self.thread.start()

    def _go(self, sched, first, count):
        try:
            self.got.append(("done", sched.submit([first, 1, 2], count, None, True,
                                                  lambda new: self.tokens.extend(new) and False)))
        except Exception as exc:             # noqa: BLE001
            self.got.append(("error", exc))

    def reply(self):
        self.thread.join(WAIT)
        return self.got[0] if self.got else None


def test_waiting_requests_start_in_arrival_order_when_streams_finish():
    dec = Decoder(room=1)
    sched = Scheduler(dec, max_streams=4)
    try:
        reqs = []
        for first in (10, 11, 12):
            reqs.append(Request(sched, first, 20))
            time.sleep(0.01)
        for r in reqs:
            kind, _ = r.reply()
            assert kind == "done" and r.tokens == [100 + i for i in range(20)]
        assert dec.admitted == [10, 11, 12]
    finally:
        sched.close()


def test_a_stream_that_gives_way_replays_and_owes_what_it_sent():
    dec = Decoder(room=2, give_way={21})
    sched = Scheduler(dec, max_streams=4)
    try:
        a = Request(sched, 20, 30)
        time.sleep(0.01)
        b = Request(sched, 21, 12)
        for r, n in ((a, 30), (b, 12)):
            kind, stats = r.reply()
            assert kind == "done", stats
            assert r.tokens == [100 + i for i in range(n)]          # each token once, in order
        assert sched.preempted == 1 and dec.admitted.count(21) == 2
    finally:
        sched.close()


def test_a_wait_past_the_limit_is_refused_with_a_capacity_error(monkeypatch):
    from tensorfold.server.errors import CapacityError

    monkeypatch.setenv(MEMORY_WAIT_ENV, "0.2")
    dec = Decoder(room=1, step_s=0.01)
    sched = Scheduler(dec, max_streams=4)
    try:
        long = Request(sched, 30, 100)                              # about a second of rounds
        time.sleep(0.02)
        late = Request(sched, 31, 5)
        kind, exc = late.reply()
        assert kind == "error" and isinstance(exc, CapacityError) and "waited" in str(exc)
        assert long.reply()[0] == "done"
    finally:
        sched.close()


def test_an_idle_decoder_still_refuses_a_request_that_cannot_fit():
    dec = Decoder(room=0)
    sched = Scheduler(dec, max_streams=4)
    try:
        kind, exc = Request(sched, 40, 5).reply()
        assert kind == "error" and isinstance(exc, NoRoom)
    finally:
        sched.close()
