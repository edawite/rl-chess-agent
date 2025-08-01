"""Utilities for computing Elo ratings.

The Elo rating system models the expected outcome of a match between two
players based on their rating difference.  This module provides a simple
function to derive an agent's rating from its results against an opponent of
known strength.
"""

from __future__ import annotations

import math
from typing import Union


def compute_elo(opponent_rating: float, wins: int, draws: int, losses: int) -> float:
    """Compute the Elo rating of the agent given match results.

    Parameters
    ----------
    opponent_rating : float
        The Elo rating of the opponent (e.g. Stockfish at a given depth).
    wins : int
        Number of games won by the agent.
    draws : int
        Number of games drawn.
    losses : int
        Number of games lost.

    Returns
    -------
    float
        The estimated Elo rating of the agent.

    Notes
    -----
    The expected score between two players A and B in Elo is given by

    .. math::

       E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}

    Inverting this formula yields the rating difference from the observed
    score.  A small epsilon is used to avoid division by zero or log of zero
    when all games are won or lost.
    """
    total_games: int = wins + draws + losses
    if total_games == 0:
        # If no games were played, assume the agent is equal to the opponent.
        return float(opponent_rating)
    # Compute the score as wins + 0.5*draws.
    score: float = wins + 0.5 * draws
    p: float = score / total_games
    # Clamp p away from 0 and 1 to avoid infinities.
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    # Compute rating difference using the inverse logistic function.
    rating_diff: float = -400.0 * math.log10((1.0 / p) - 1.0)
    return opponent_rating + rating_diff
