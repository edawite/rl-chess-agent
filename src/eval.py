"""Evaluation script for trained PPO chess agents against Stockfish.

This module provides a command‑line interface to assess the strength of a
trained model by pitting it against Stockfish at a configurable depth.
The results are summarised as win/draw/loss counts and an estimated Elo
rating.  When run as a script it writes a CSV entry to the ``results``
directory and prints a concise report to stdout.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import yaml
import chess
import chess.engine
import numpy as np
from stable_baselines3 import PPO

from .envs.chess_env import ChessEnv
from .elo import compute_elo
from .utils.logger import CSVLogger


def play_against_stockfish(
    model: PPO, engine: chess.engine.SimpleEngine, depth: int, games: int
) -> Dict[str, int]:
    """Play a series of games against Stockfish and record outcomes.

    The agent is assigned colours at random in each game.  Games are played
    until completion under the rules of chess.

    Parameters
    ----------
    model : PPO
        A trained stable‑baselines3 PPO model.
    engine : chess.engine.SimpleEngine
        A running Stockfish engine instance.
    depth : int
        Search depth used for Stockfish moves.
    games : int
        Number of games to play.

    Returns
    -------
    dict
        A dictionary with keys ``"wins"``, ``"draws"`` and ``"losses"``.
    """
    results = {"wins": 0, "draws": 0, "losses": 0}
    env = ChessEnv()
    # Precompute limit to avoid object creation in loop.
    limit = chess.engine.Limit(depth=depth)
    for _ in range(games):
        obs, info = env.reset()
        done = False
        # Randomly choose which side the agent will play.
        agent_is_white = bool(random.getrandbits(1))
        while not done:
            # Determine whose turn it is and let the appropriate player move.
            if env.board.turn == chess.WHITE:
                if agent_is_white:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    # Update observation after agent move.
                    continue
                # Stockfish plays as white.
                result = engine.play(env.board, limit)
                env.board.push(result.move)
                # Reset shaping state after external move.
                env._last_material = env._material_diff()
                obs = env._get_obs()
                # Check for termination after Stockfish move.
                if env.board.is_game_over():
                    break
            else:
                # Black to move.
                if not agent_is_white:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    continue
                # Stockfish plays as black.
                result = engine.play(env.board, limit)
                env.board.push(result.move)
                env._last_material = env._material_diff()
                obs = env._get_obs()
                if env.board.is_game_over():
                    break
        # Record outcome from agent's perspective.
        outcome = env.board.result()
        if outcome == "1-0":
            results["wins" if agent_is_white else "losses"] += 1
        elif outcome == "0-1":
            results["losses" if agent_is_white else "wins"] += 1
        else:
            results["draws"] += 1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO chess agent against Stockfish")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to the YAML configuration file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (zip file)")
    parser.add_argument("--stockfish-depth", type=int, default=None, help="Override Stockfish search depth")
    parser.add_argument("--games", type=int, default=None, help="Override number of evaluation games")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write a CSV summary")
    args = parser.parse_args()

    # Load configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    depth = args.stockfish_depth if args.stockfish_depth is not None else config.get("stockfish_depth", 2)
    games = args.games if args.games is not None else config.get("eval_games", 20)
    stockfish_path = config.get("stockfish_path")
    if not stockfish_path or not Path(stockfish_path).exists():
        raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}")
    # Load trained model.
    model = PPO.load(args.model_path)
    # Spin up Stockfish engine.
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        results = play_against_stockfish(model, engine, depth=depth, games=games)
    finally:
        engine.quit()
    wins, draws, losses = results["wins"], results["draws"], results["losses"]
    # Baseline ratings by depth.  These values are heuristics and may differ
    # from official Stockfish ratings.
    baseline_by_depth = {1: 1000, 2: 1300, 3: 1600, 4: 1800}
    opponent_rating = baseline_by_depth.get(depth, 1300)
    elo = compute_elo(opponent_rating, wins, draws, losses)
    print(f"Evaluation complete: {games} games at depth {depth}")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Estimated Elo: {elo:.1f} vs Stockfish depth {depth}")
    # Optionally write results to CSV.
    if args.output:
        output_path = Path(args.output)
        with CSVLogger(output_path, headers=["model_path", "stockfish_depth", "games", "wins", "draws", "losses", "elo"]) as logger:
            logger.log([args.model_path, depth, games, wins, draws, losses, elo])


if __name__ == "__main__":
    main()
