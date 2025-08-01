"""Tests for the Elo rating computation."""

import unittest
import sys
from pathlib import Path

# Make the `src` package importable when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.elo import compute_elo


class TestElo(unittest.TestCase):
    def test_monotonic(self) -> None:
        base = 1200.0
        elo_win = compute_elo(base, wins=3, draws=0, losses=0)
        elo_draw = compute_elo(base, wins=0, draws=3, losses=0)
        elo_loss = compute_elo(base, wins=0, draws=0, losses=3)
        self.assertGreater(elo_win, elo_draw)
        self.assertGreater(elo_draw, elo_loss)

    def test_no_games(self) -> None:
        base = 1500.0
        elo_same = compute_elo(base, wins=0, draws=0, losses=0)
        self.assertEqual(elo_same, base)


if __name__ == "__main__":
    unittest.main()
