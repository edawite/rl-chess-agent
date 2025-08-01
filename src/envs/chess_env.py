"""Gymnasium environment for self‑play chess using python‑chess.

This module defines a single‑agent environment where a policy controls both
sides of a chess game in alternation.  Each call to :meth:`step` applies one
move for the current player.  Illegal moves are penalised and terminate the
episode to discourage the agent from playing invalid moves.

The observation consists of 13 planes of shape 8×8:

* six planes for white pieces (pawn, knight, bishop, rook, queen, king);
* six planes for black pieces;
* one plane indicating whose turn it is (all ones when it is white's move).

These planes are flattened into a one‑dimensional array for consumption by
stable‑baselines3.  Reward shaping is configurable via the ``reward_config``
dictionary; by default only the game outcome contributes to the reward.  An
optional material difference term can be added to provide sparse feedback
throughout the game.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import chess
from typing import Optional, Dict, Any, Tuple


class ChessEnv(gym.Env[np.ndarray, int]):
    """A single‑agent self‑play chess environment.

    The agent controls whichever side is to move.  The action space enumerates
    all possible chess moves as triples ``(from_square, to_square, promotion)``.
    Only a small subset of these actions are legal in any given position; if
    the agent selects an illegal action the environment will apply a fixed
    penalty and terminate the episode.

    Parameters
    ----------
    board_fen : str or None, optional
        Starting position in Forsyth‑Edwards Notation.  When ``None`` the
        standard chess starting position is used.
    reward_config : dict, optional
        Coefficients for reward shaping.  Expected keys include ``"win"``,
        ``"draw"``, ``"loss"``, ``"material"`` and ``"checkmate_bonus"``.
    illegal_move_penalty : float, default ``-1.0``
        Reward assigned when an illegal move is attempted.  A negative value
        encourages the agent to learn the set of legal moves.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        board_fen: Optional[str] = None,
        reward_config: Optional[Dict[str, float]] = None,
        illegal_move_penalty: float = -1.0,
    ) -> None:
        super().__init__()
        # Keep a copy of the initial position for deterministic resets.
        self._initial_fen: Optional[str] = board_fen
        self.board: chess.Board = chess.Board(board_fen) if board_fen else chess.Board()
        # Reward parameters with sensible defaults.
        self.reward_config: Dict[str, float] = reward_config or {
            "win": 1.0,
            "draw": 0.0,
            "loss": -1.0,
            "material": 0.0,
            "checkmate_bonus": 0.0,
        }
        self.illegal_move_penalty: float = illegal_move_penalty

        # Precompute the action lookup table.  Each entry is a tuple
        # (from_square, to_square, promotion) that can be passed to
        # :class:`chess.Move` to construct a move.  Promotions include
        # ``None`` for no promotion and the four common promotion piece types.
        promotions = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        self._actions: list[Tuple[int, int, Optional[int]]] = []
        for from_sq in range(64):
            for to_sq in range(64):
                for promo in promotions:
                    self._actions.append((from_sq, to_sq, promo))

        # Define the Gymnasium spaces.  Observations are flattened arrays of
        # length 13×8×8=832 with values in {0,1}.  The action space covers all
        # precomputed moves.
        self.action_space: spaces.Discrete = spaces.Discrete(len(self._actions))
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(13 * 8 * 8,), dtype=np.float32
        )

        # Track the material balance from the current player's perspective.
        self._last_material: float = self._material_diff()

    # -- Utility functions -----------------------------------------------------

    def _material_diff(self) -> float:
        """Return the material balance from the side‑to‑move's perspective.

        Positive values favour the current player; negative values favour
        the opponent.  Piece values follow common chess heuristics.
        """
        values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }
        white = sum(
            values[piece.piece_type]
            for piece in self.board.piece_map().values()
            if piece.color == chess.WHITE
        )
        black = sum(
            values[piece.piece_type]
            for piece in self.board.piece_map().values()
            if piece.color == chess.BLACK
        )
        # Perspective: positive means "better for side to move"
        return (white - black) if self.board.turn == chess.WHITE else (black - white)

    def _get_obs(self) -> np.ndarray:
        """Convert the current board state into a flattened observation."""
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        for square, piece in self.board.piece_map().items():
            # Planes 0‑5 are white pieces, 6‑11 are black pieces.
            plane_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            planes[plane_idx, row, col] = 1.0
        # Plane 12 indicates whose turn it is: ones for white, zeros for black.
        if self.board.turn == chess.WHITE:
            planes[12, :, :] = 1.0
        else:
            planes[12, :, :] = 0.0
        return planes.flatten()

    def legal_moves_mask(self) -> np.ndarray:
        """Return a boolean mask over the action space indicating legal moves.

        This mask is useful for evaluation and analysis; stable‑baselines3 does
        not currently consume action masks directly.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        promotions = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for move in self.board.legal_moves:
            promo_idx = 0
            if move.promotion is not None:
                promo_idx = promotions.index(move.promotion)
            index = (move.from_square * 64 + move.to_square) * len(promotions) + promo_idx
            mask[index] = True
        return mask

    # -- Gymnasium API --------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial position.

        Returns a fresh observation and an empty ``info`` dictionary.
        """
        super().reset(seed=seed)
        self.board = chess.Board(self._initial_fen) if self._initial_fen else chess.Board()
        self._last_material = self._material_diff()
        observation = self._get_obs()
        info: Dict[str, Any] = {}
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one move to the board and return the new observation and reward.

        If the action is illegal the environment applies a fixed penalty and
        terminates the episode.  Otherwise the reward is the shaped difference
        between the new and previous material balance, plus the outcome when
        the game concludes.

        Returns
        -------
        observation : ndarray
            The flattened board representation after the move.
        reward : float
            The reward for the selected move.
        terminated : bool
            ``True`` if the game has ended.
        truncated : bool
            ``True`` if the episode has been truncated for any reason (unused).
        info : dict
            Additional diagnostic information.
        """
        from_sq, to_sq, promo = self._actions[action]
        move = chess.Move(from_sq, to_sq, promo)
        reward = 0.0
        terminated = False
        info: Dict[str, Any] = {}

        # Penalise and terminate on illegal moves.
        if move not in self.board.legal_moves:
            reward = float(self.illegal_move_penalty)
            terminated = True
            info["illegal_move"] = True
            observation = self._get_obs()
            return observation, reward, terminated, False, info

        # Apply the move.
        self.board.push(move)

        # Outcome reward if the game is over.
        if self.board.is_game_over():
            result = self.board.result()  # '1-0', '0-1' or '1/2-1/2'
            if result == "1-0":
                base = self.reward_config.get("win", 1.0)
            elif result == "0-1":
                base = self.reward_config.get("loss", -1.0)
            else:
                base = self.reward_config.get("draw", 0.0)
            reward += base
            # Optional checkmate bonus.
            if self.board.is_checkmate():
                reward += self.reward_config.get("checkmate_bonus", 0.0)
            terminated = True
        else:
            # Material difference shaping.
            new_material = self._material_diff()
            reward += self.reward_config.get("material", 0.0) * (new_material - self._last_material)
            self._last_material = new_material

        observation = self._get_obs()
        return observation, reward, terminated, False, info

    # -- Rendering ------------------------------------------------------------

    def render(self, mode: str = "human") -> str:
        """Render the current board position.

        When ``mode`` is ``"human"`` or ``"ansi"`` this method returns a
        unicode diagram of the board.  Other modes are not supported.
        """
        if mode in ("human", "ansi"):
            return str(self.board)
        raise NotImplementedError(f"Unsupported render mode: {mode}")
