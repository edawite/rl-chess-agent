"""Train a PPO self‑play chess agent using stable‑baselines3.

This script reads hyperparameters and environment settings from a YAML
configuration file and trains a Proximal Policy Optimisation (PPO) agent
via self‑play.  After training, the model is saved to disk and evaluated
against Stockfish.  Training rewards and evaluation statistics are logged
to CSV files in the ``results`` directory.  Plots of the reward and Elo
progression can also be generated.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import yaml
import numpy as np
import torch
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs.chess_env import ChessEnv
from .eval import play_against_stockfish
from .elo import compute_elo
from .utils.logger import CSVLogger
from .utils.plotting import (
    plot_training_reward,
    plot_elo_curve,
    plot_win_draw_loss,
)

import chess  # required for Stockfish evaluation


def set_global_seeds(seed: int) -> None:
    """Set seeds for random number generators across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(config: dict, seed: int) -> DummyVecEnv:
    """Create a vectorised chess environment with monitoring.

    Each environment instance is wrapped in a :class:`Monitor` to record
    episode rewards.  The number of parallel environments is taken from
    ``config['n_envs']``.
    """

    def make_single_env() -> Monitor:
        env = ChessEnv(
            board_fen=config.get("board_fen"),
            reward_config=config.get("reward"),
            illegal_move_penalty=config.get("illegal_move_penalty", -1.0),
        )
        # Wrap with Monitor to track episode returns.  Filename is None to
        # store logs in memory; they can be accessed via get_episode_rewards().
        return Monitor(env)

    n_envs = int(config.get("n_envs", 1))
    env_fns = [make_single_env for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(seed)
    return vec_env


def train_and_evaluate(config_path: Path | str, override_seeds: List[int] | None = None) -> None:
    """Run training for each seed defined in the configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    override_seeds : list of int, optional
        If provided, these seeds override the values in the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    seeds: List[int] = override_seeds or list(config.get("seeds", [0]))
    # Prepare results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    # Prepare logs for cumulative Elo across seeds
    elo_log_path = results_dir / "elo_log.csv"
    with CSVLogger(elo_log_path, headers=["seed", "episode", "wins", "draws", "losses", "elo"]) as elo_logger:
        for seed in seeds:
            print(f"\n===== Training with seed {seed} =====")
            set_global_seeds(seed)
            # Build environment
            env = build_env(config, seed)
            # Instantiate the PPO agent
            model = PPO(
                policy="MlpPolicy",
                env=env,
                gamma=float(config.get("gamma", 0.99)),
                gae_lambda=float(config.get("gae_lambda", 0.95)),
                vf_coef=float(config.get("vf_coef", 0.5)),
                ent_coef=float(config.get("ent_coef", 0.01)),
                learning_rate=float(config.get("lr", 3e-4)),
                seed=seed,
                verbose=1,
            )
            total_timesteps = int(config.get("total_timesteps", 1000000))
            model.learn(total_timesteps=total_timesteps)
            # Save the trained model
            model_path = results_dir / f"ppo_chess_seed{seed}.zip"
            model.save(model_path)
            # Extract episode returns from the Monitor wrappers
            # Only the first environment's monitor contains returns for all episodes.
            episode_returns: List[float] = env.envs[0].get_episode_rewards()
            # Write training reward log
            train_log_path = results_dir / f"train_log_seed{seed}.csv"
            with CSVLogger(train_log_path, headers=["episode", "reward"]) as train_logger:
                for episode_idx, reward in enumerate(episode_returns):
                    train_logger.log([episode_idx, reward])
            # Evaluate the trained model against Stockfish
            stockfish_path = config.get("stockfish_path")
            if not stockfish_path or not Path(stockfish_path).exists():
                raise FileNotFoundError(
                    f"Stockfish binary not found at path: {stockfish_path}. Please update the config."
                )
            depth = int(config.get("stockfish_depth", 2))
            games = int(config.get("eval_games", 20))
            # Start Stockfish engine
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            try:
                results = play_against_stockfish(model, engine, depth=depth, games=games)
            finally:
                engine.quit()
            wins, draws, losses = results["wins"], results["draws"], results["losses"]
            # Estimate Elo based on a heuristic baseline per depth
            baseline_by_depth = {1: 1000, 2: 1300, 3: 1600, 4: 1800}
            opponent_rating = baseline_by_depth.get(depth, 1300)
            elo = compute_elo(opponent_rating, wins, draws, losses)
            # Log the evaluation result
            elo_logger.log([seed, len(episode_returns), wins, draws, losses, elo])
            # Generate plots for this seed
            plot_training_reward(train_log_path, results_dir / f"reward_curve_seed{seed}.png")
            # For a single evaluation at the end, the Elo curve is just one point;
            # nonetheless we write it for completeness.
            plot_elo_curve(elo_log_path, results_dir / f"elo_curve.png")
            plot_win_draw_loss(elo_log_path, results_dir / f"wld_curve.png")
            print(f"Seed {seed}: wins={wins}, draws={draws}, losses={losses}, Elo={elo:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO chess agent via self‑play")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--seeds", type=int, nargs="*", help="Override the seeds from config")
    args = parser.parse_args()
    seed_override = args.seeds if args.seeds and len(args.seeds) > 0 else None
    train_and_evaluate(args.config, override_seeds=seed_override)


if __name__ == "__main__":
    main()
