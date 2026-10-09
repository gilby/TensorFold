"""Each rank's share of DFlash2's draft vocabulary: in order, covering it once, two or four ranks (no GPU)."""

import pytest

pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda.dflash2 import _rank_spans  # noqa: E402

SPANS = ((0, 98304), (248032, 248320))


@pytest.mark.parametrize("world", [2, 4])
def test_ranks_cover_the_draft_vocabulary_once_in_order(world):
    parts = [_rank_spans(SPANS, r, world) for r in range(world)]
    ids = [i for p in parts for a, b in p for i in range(a, b)]
    assert ids == [i for a, b in SPANS for i in range(a, b)]
    share = -(-sum(b - a for a, b in SPANS) // world)
    assert [sum(b - a for a, b in p) for p in parts[:-1]] == [share] * (world - 1)


def test_two_ranks_keep_their_halves():
    assert [_rank_spans(SPANS, r) for r in (0, 1)] == [_rank_spans(SPANS, r, 2) for r in (0, 1)]
