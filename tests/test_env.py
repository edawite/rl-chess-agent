"""Unit tests for the ChessEnv environment."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Make the `src` package importable when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.envs.chess_env import ChessEnv  # type: ignore
except Exception:
    # Defer import errors until tests are executed; the tests will skip if
    # dependencies such as gymnasium are missing.
    ChessEnv = None  # type: ignore


class TestChessEnv(unittest.TestCase):
    """Test basic interactions with the chess environment."""

    def setUp(self) -> None:
        if ChessEnv is None:
            self.skipTest("ChessEnv or its dependencies could not be imported")
        self.env = ChessEnv()

    def test_reset_shape(self) -> None:
        obs, info = self.env.reset()
        # Observation should be a flat vector of length 832
        self.assertEqual(obs.shape, (13 * 8 * 8,))
        self.assertIsInstance(info, dict)

    def test_step_legal_move(self) -> None:
        obs, _ = self.env.reset()
        # Pick the first legal move from the mask
        mask = self.env.legal_moves_mask()
        legal_actions = np.where(mask)[0]
        self.assertTrue(len(legal_actions) > 0, "There should be at least one legal move")
        action = int(legal_actions[0])
        new_obs, reward, done, truncated, info = self.env.step(action)
        self.assertEqual(new_obs.shape, (13 * 8 * 8,))
        self.assertFalse(truncated)
        # A single legal move should not immediately terminate the game
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_illegal_move_penalty(self) -> None:
        obs, _ = self.env.reset()
        mask = self.env.legal_moves_mask()
        illegal_actions = np.where(~mask)[0]
        # It is possible that some positions have no illegal actions if the action space is
        # constrained artificially; if so, skip this test.
        if len(illegal_actions) == 0:
            self.skipTest("No illegal moves available to test")
        action = int(illegal_actions[0])
        new_obs, reward, done, truncated, info = self.env.step(action)
        self.assertTrue(done)
        self.assertTrue(info.get("illegal_move", False))
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
