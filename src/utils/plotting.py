"""Plotting utilities for visualising training and evaluation metrics.

This module leverages Matplotlib to produce simple line and bar charts for
reward progression, Elo trends and win/draw/loss ratios.  These plots are
saved to disk rather than displayed interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_reward(csv_path: Path | str, save_path: Path | str) -> None:
    """Plot cumulative reward per episode from a training log.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing training logs with columns including
        ``episode`` and ``reward``.
    save_path : str or Path
        Path where the generated PNG will be saved.  Parent directories
        are created as needed.
    """
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "reward" not in df.columns:
        raise ValueError("CSV does not contain required columns 'episode' and 'reward'")
    plt.figure(figsize=(8, 4))
    plt.plot(df["episode"], df["reward"], label="Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Reward Over Episodes")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_elo_curve(csv_path: Path | str, save_path: Path | str) -> None:
    """Plot Elo rating versus episode index from a training/evaluation log.

    The CSV is expected to contain ``episode`` and ``elo`` columns.  If
    ``episode`` is missing the index of the DataFrame is used instead.
    """
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    df = pd.read_csv(csv_path)
    episodes = df["episode"] if "episode" in df.columns else df.index
    if "elo" not in df.columns:
        raise ValueError("CSV does not contain required column 'elo'")
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, df["elo"], color="tab:blue")
    plt.xlabel("Episode")
    plt.ylabel("Elo Rating")
    plt.title("Estimated Elo Over Training")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_win_draw_loss(csv_path: Path | str, save_path: Path | str) -> None:
    """Plot stacked win/draw/loss percentages from evaluation logs.

    The CSV file must include columns ``episode``, ``wins``, ``draws`` and
    ``losses``.  The values are normalised per episode to compute
    percentages.
    """
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    df = pd.read_csv(csv_path)
    required = {"episode", "wins", "draws", "losses"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    total_games = df[["wins", "draws", "losses"]].sum(axis=1).astype(float)
    # Avoid division by zero
    total_games[total_games == 0] = 1.0
    win_pct = df["wins"] / total_games
    draw_pct = df["draws"] / total_games
    loss_pct = df["losses"] / total_games
    plt.figure(figsize=(8, 4))
    plt.stackplot(
        df["episode"],
        win_pct,
        draw_pct,
        loss_pct,
        labels=["Win", "Draw", "Loss"],
        colors=["tab:green", "tab:gray", "tab:red"],
        alpha=0.8,
    )
    plt.xlabel("Episode")
    plt.ylabel("Proportion")
    plt.title("Win/Draw/Loss Ratio Over Time")
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
