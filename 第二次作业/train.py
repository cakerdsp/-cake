from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm, ListedColormap

from algorithms import run_q_learning, run_sarsa
from cliff_walk_env import CliffWalkEnv, State, seed_everything

# 设置中文字体，兼容常见操作系统
rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Songti SC",
    "WenQuanYi Micro Hei",
]
rcParams["axes.unicode_minus"] = False


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return data
    cumulative = np.cumsum(np.insert(data, 0, 0))
    result = (cumulative[window_size:] - cumulative[:-window_size]) / window_size
    # 为了保持长度一致，前面补值
    prefix = np.full(window_size - 1, result[0])
    return np.concatenate([prefix, result])


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def plot_cliffwalk_layout(
    env: CliffWalkEnv, output_path: str | Path = "figures/cliffwalk_layout.png"
) -> None:
    ensure_dir(Path(output_path).parent)
    grid = np.zeros((env.n_rows, env.n_cols))
    cliff_states = env.cliff_states

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state = (r, c)
            if state in cliff_states:
                grid[r, c] = 1

    sr, sc = env.start_state
    gr, gc = env.terminal_state
    grid[sr, sc] = 2
    grid[gr, gc] = 3

    cmap = ListedColormap(
        ["#f5f5f5", "#404040", "#2e8b57", "#d4a017"], name="cliffwalk_layout"
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")

    ax.set_xticks(np.arange(env.n_cols))
    ax.set_yticks(np.arange(env.n_rows))
    ax.set_xticklabels(np.arange(env.n_cols))
    ax.set_yticklabels(np.arange(env.n_rows))
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.8)
    ax.tick_params(which="both", length=0)

    ax.text(
        sc,
        sr,
        "S",
        color="white",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        gc,
        gr,
        "G",
        color="black",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.set_title("Cliff Walk 地图结构")
    ax.set_xlabel("列索引")
    ax.set_ylabel("行索引")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_policy(
    env: CliffWalkEnv,
    q_values: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    ensure_dir(Path(output_path).parent)

    base_grid = np.zeros((env.n_rows, env.n_cols))
    cliff_states = env.cliff_states

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if (r, c) in cliff_states:
                base_grid[r, c] = 1

    sr, sc = env.start_state
    gr, gc = env.terminal_state
    base_grid[sr, sc] = 2
    base_grid[gr, gc] = 3

    cmap = ListedColormap(
        ["#f5f5f5", "#404040", "#2e8b57", "#d4a017"], name="cliffwalk_policy"
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(base_grid, cmap=cmap, norm=norm, origin="upper")

    ax.set_xticks(np.arange(env.n_cols))
    ax.set_yticks(np.arange(env.n_rows))
    ax.set_xticklabels(np.arange(env.n_cols))
    ax.set_yticklabels(np.arange(env.n_rows))
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.8)
    ax.tick_params(which="both", length=0)

    action_symbols = {0: "^", 1: ">", 2: "v", 3: "<"}

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state: State = (r, c)
            if state in cliff_states:
                ax.text(
                    c,
                    r,
                    "X",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )
                continue
            if state == env.start_state:
                ax.text(
                    c,
                    r,
                    "S",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
                continue
            if state == env.terminal_state:
                ax.text(
                    c,
                    r,
                    "G",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
                continue

            state_idx = env.state_to_index(state)
            q_state = q_values[state_idx]
            if np.allclose(q_state, 0.0):
                symbol = "·"
            else:
                action = int(np.argmax(q_state))
                symbol = action_symbols.get(action, ".")
            ax.text(
                c,
                r,
                symbol,
                color="#1f77b4",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_title(title)
    ax.set_xlabel("列索引")
    ax.set_ylabel("行索引")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    seed_everything(42)
    env = CliffWalkEnv()

    config_sarsa: Dict[str, float | int] = {
        "episodes": 500,
        "alpha": 0.2,
        "gamma": 1.0,
        "epsilon": 0.2,
        "epsilon_decay": 0.98,
        "epsilon_min": 0.01,
    }

    config_q: Dict[str, float | int] = {
        "episodes": 500,
        "alpha": 0.2,
        "gamma": 1.0,
        "epsilon": 0.2,
        "epsilon_decay": 0.98,
        "epsilon_min": 0.01,
    }

    sarsa_stats = run_sarsa(env, **config_sarsa)
    seed_everything(42)  # 重新设定种子确保公平比较
    q_stats = run_q_learning(env, **config_q)

    plot_cliffwalk_layout(env)
    plot_results(sarsa_stats.episode_rewards, q_stats.episode_rewards)
    plot_episode_lengths(sarsa_stats.episode_lengths, q_stats.episode_lengths)
    plot_policy(
        env,
        sarsa_stats.q_values,
        "SARSA 贪婪策略示意图",
        "figures/cliffwalk_policy_sarsa.png",
    )
    plot_policy(
        env,
        q_stats.q_values,
        "Q-learning 贪婪策略示意图",
        "figures/cliffwalk_policy_qlearning.png",
    )


def plot_results(sarsa_rewards, q_rewards) -> None:
    episodes = np.arange(1, len(sarsa_rewards) + 1)
    ma_window = 10
    sarsa_ma = moving_average(np.array(sarsa_rewards), ma_window)
    q_ma = moving_average(np.array(q_rewards), ma_window)

    ensure_dir("figures")
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, sarsa_rewards, alpha=0.2, label="SARSA (原始回报)")
    plt.plot(episodes, q_rewards, alpha=0.2, label="Q-learning (原始回报)")
    plt.plot(episodes, sarsa_ma, label=f"SARSA (滑动平均 {ma_window})", linewidth=2)
    plt.plot(episodes, q_ma, label=f"Q-learning (滑动平均 {ma_window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return (累计奖励)")
    plt.title("Cliff Walk：SARSA vs Q-learning 回报对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/cliffwalk_returns.png", dpi=300)
    plt.close()


def plot_episode_lengths(sarsa_lengths, q_lengths) -> None:
    episodes = np.arange(1, len(sarsa_lengths) + 1)
    ma_window = 10
    sarsa_ma = moving_average(np.array(sarsa_lengths), ma_window)
    q_ma = moving_average(np.array(q_lengths), ma_window)

    ensure_dir("figures")
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, sarsa_lengths, alpha=0.2, label="SARSA (原始步数)")
    plt.plot(episodes, q_lengths, alpha=0.2, label="Q-learning (原始步数)")
    plt.plot(
        episodes,
        sarsa_ma,
        label=f"SARSA 步数 (滑动平均 {ma_window})",
        linewidth=2,
    )
    plt.plot(
        episodes,
        q_ma,
        label=f"Q-learning 步数 (滑动平均 {ma_window})",
        linewidth=2,
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (步数)")
    plt.title("Cliff Walk：每回合步数对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/cliffwalk_lengths.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()


