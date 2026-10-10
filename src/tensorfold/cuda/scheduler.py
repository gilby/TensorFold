"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.
# Requests memory can't hold yet wait in arrival order (MEMORY_WAIT_ENV caps the wait); one that gave way replays first.

from __future__ import annotations

import collections
import itertools
import os
import queue
import threading
import time
from typing import Any, Callable

from .memory_gate import NoRoom
from .streams import Stream

MEMORY_WAIT_ENV = "TENSORFOLD_MEMORY_WAIT_S"
MEMORY_WAIT_S = 900.0            # longest a request waits for memory before it is refused
RETRY_S = 1.0                    # a waiting request is asked again this often when no stream has finished


def memory_wait_s() -> float:
    value = os.environ.get(MEMORY_WAIT_ENV, "").strip()
    if not value:
        return MEMORY_WAIT_S
    try:
        seconds = float(value)
    except ValueError:
        raise ValueError(f"{MEMORY_WAIT_ENV}: seconds, not {value!r}") from None
    if not seconds > 0:
        raise ValueError(f"{MEMORY_WAIT_ENV}: a positive number of seconds, not {value!r}")
    return seconds


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def put_many(self, items) -> None:
        """Publish a group before waking the worker, preserving arrival and background priority."""

        with self.not_empty:
            for item in items:
                self._put((1 if item[0].background else 0, next(self._order), item))
                self.unfinished_tasks += 1
            self.not_empty.notify()

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        super().put((2, next(self._order), None))

    def foreground(self) -> bool:
        """Whether a foreground request waits."""

        with self.mutex:
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        self.waiting = Waiting()
        # requests waiting for memory, oldest first, admitted before any other: (stream, box, since)
        self.parked: collections.deque = collections.deque()
        self.wait_s = memory_wait_s()
        self.retry_at = 0.0                          # when the first parked request is asked again
        self.boxes: dict[int, queue.Queue] = {}
        self.yields = 0                              # background streams that gave up their lane
        self.preempted = 0                           # streams that gave way for memory and replay
        if hasattr(decoder, "arrived"):              # a decoder filling prompts lets a new request in between passes
            decoder.arrived = self.waiting.foreground
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop the worker once idle and let go of the decoder (its thread held it, and its weights, until now)."""

        self.waiting.stop()
        self.thread.join()
        self.decoder = None

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, probabilities: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities)
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def decide(self, decision: Any) -> dict:
        """Prefill a decision prompt (``families.qwen3_5.cuda.decisions``) in turn with the streams; its stats."""

        box: queue.Queue = queue.Queue()
        self.waiting.put((decision, box))
        kind, value = box.get()
        if kind == "error":
            raise value
        return value

    def submit_many(self, requests: list[dict]) -> list[dict]:
        """Queue isolated one-shot requests atomically and return ordered stats, draining errors too."""

        boxes = []
        for r in requests:
            box: queue.Queue = queue.Queue()
            stream = Stream(list(r["prompt"]), max(1, r["count"]), r["sampling"], draft=r.get("draft", True),
                            stop_eos=r.get("stop_eos", True), probabilities=r.get("probabilities"))
            stream.emit = lambda new: False
            boxes.append((stream, box))
        self.waiting.put_many(boxes)
        results, error = [], None
        for _, box in boxes:
            while True:
                kind, value = box.get()
                if kind == "tokens":
                    continue
                if kind == "error":
                    error = error or value
                    results.append(None)
                else:
                    results.append(value)
                break
        if error is not None:
            raise error
        return results

    def _expire(self) -> None:
        """Refuse parked requests that have waited ``wait_s``: a clear error now rather than a hung client."""

        from tensorfold.server.errors import CapacityError

        now = time.monotonic()
        while self.parked and now - self.parked[0][2] > self.wait_s:
            stream, box, since = self.parked.popleft()
            box.put(("error", CapacityError(
                f"this request waited {now - since:.0f} s for GPU memory while {self.decoder.live()} streams decoded "
                f"({MEMORY_WAIT_ENV}={self.wait_s:g}); retry later, or shorten the prompt or max_tokens")))

    def _admit(self, first=None) -> list[Stream]:
        done = []
        self._expire()
        while self.decoder.live() < self.max_streams:
            parked = False
            if first is not None:
                (stream, box), first = first, None
                since = time.monotonic()
            elif self.parked:
                if time.monotonic() < self.retry_at:
                    break                            # nothing freed since it was refused: ask again later
                (stream, box, since), parked = self.parked[0], True
            else:
                try:
                    stream, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
                since = time.monotonic()
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom as exc:
                self.boxes.pop(id(stream))
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    if not parked:
                        self.parked.append((stream, box, since))
                    self.retry_at = time.monotonic() + RETRY_S
                    break
                if parked:
                    self.parked.popleft()
                box.put(("error", exc))
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                if parked:
                    self.parked.popleft()
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if parked:
                self.parked.popleft()
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        self.waiting.put((stream.continued(), box))

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and not self.parked
            first = self.waiting.get() if idle else None                              # idle: wait for a request
            if idle and first is None:
                return                                                                # close()
            done = self._admit(first)
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            if done:
                self.retry_at = 0.0                  # a stream finished: a parked request may fit now
            replays = []
            for s in done:
                if s.preempted and s.error is None:
                    box = self.boxes.pop(id(s), None)
                    if box is not None:              # its reply so far is owed, not resent: it replays first
                        replays.append((s.continued(), box, time.monotonic()))
                        self.preempted += 1
                        print(f"[tensorfold] a stream gave way for memory after {len(s.out) + len(s.owed)} tokens; it "
                              "replays from its prompt when a stream finishes", flush=True)
                    continue
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
            self.parked.extendleft(reversed(replays))
