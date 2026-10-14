"""The memory gate's reserve: 2 GiB beside a unified GPU, a measured floor on a discrete card, or as configured."""

import pytest

from tensorfold.cuda.memory_gate import (DISCRETE_RESERVE, RESERVE_ENV, UNIFIED_RESERVE, MemoryGate,
                                         reserve_bytes)


def test_unified_memory_keeps_two_gib_and_is_not_remeasured(monkeypatch):
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    assert reserve_bytes(0, unified=True) == (UNIFIED_RESERVE, False) == (2 * 1024**3, False)
    assert reserve_bytes(3 * 1024**3, unified=True) == (3 * 1024**3, False)


def test_a_discrete_card_starts_from_its_floor_and_is_measured(monkeypatch):
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    assert reserve_bytes(0, unified=False) == (DISCRETE_RESERVE, True)
    assert DISCRETE_RESERVE < UNIFIED_RESERVE
    assert reserve_bytes(1024**3, unified=False) == (1024**3, True)


def test_the_environment_sets_it_exactly(monkeypatch):
    monkeypatch.setenv(RESERVE_ENV, "768")
    assert reserve_bytes(1 << 40, unified=True) == (768 * 1024**2, False)
    monkeypatch.setenv(RESERVE_ENV, "1.5")
    with pytest.raises(ValueError, match="whole MiB"):
        reserve_bytes(0, unified=False)


def test_the_gate_remembers_its_peak():
    g = MemoryGate(100, 0)
    g.take(40)
    g.take(30)
    g.give(50)
    g.take(10)
    assert g.held == 30 and g.peak == 70
