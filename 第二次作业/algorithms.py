"""
SARSA 与 Q-learning 算法实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from cliff_walk_env import Action, CliffWalkEnv, State


Policy = Callable[[np.ndarray, int, float], Action]


@dataclass
class EpisodeStats:
    episode_rewards: List[float]
    episode_lengths: List[int]
    q_values: np.ndarray


def epsilon_greedy_policy(q_values: np.ndarray, n_actions: int, epsilon: float) -> Action:
    """基于 ε-greedy 选择动作。"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(q_values))


def run_sarsa(
    env: CliffWalkEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
) -> EpisodeStats:
    """执行 SARSA 训练。"""
    n_states = env.n_states
    n_actions = env.n_actions
    q = np.zeros((n_states, n_actions), dtype=float)
    rewards: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        action = epsilon_greedy_policy(q[state_idx], n_actions, epsilon)

        total_reward = 0.0
        step_count = 0
        terminated = False

        while not terminated:
            transition = env.step(action)
            next_state_idx = env.state_to_index(transition.next_state)
            next_action = epsilon_greedy_policy(q[next_state_idx], n_actions, epsilon)

            td_target = transition.reward + gamma * q[next_state_idx, next_action]
            td_error = td_target - q[state_idx, action]
            q[state_idx, action] += alpha * td_error

            total_reward += transition.reward
            step_count += 1

            state_idx = next_state_idx
            action = next_action
            terminated = transition.terminated

            # 避免无限循环
            if step_count > 10_000:
                break

        rewards.append(total_reward)
        lengths.append(step_count)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return EpisodeStats(rewards, lengths, q)


def run_q_learning(
    env: CliffWalkEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
) -> EpisodeStats:
    """执行 Q-learning 训练。"""
    n_states = env.n_states
    n_actions = env.n_actions
    q = np.zeros((n_states, n_actions), dtype=float)
    rewards: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)

        total_reward = 0.0
        step_count = 0
        terminated = False

        while not terminated:
            action = epsilon_greedy_policy(q[state_idx], n_actions, epsilon)
            transition = env.step(action)
            next_state_idx = env.state_to_index(transition.next_state)

            td_target = transition.reward + gamma * np.max(q[next_state_idx])
            td_error = td_target - q[state_idx, action]
            q[state_idx, action] += alpha * td_error

            total_reward += transition.reward
            step_count += 1
            state_idx = next_state_idx
            terminated = transition.terminated

            if step_count > 10_000:
                break

        rewards.append(total_reward)
        lengths.append(step_count)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return EpisodeStats(rewards, lengths, q)


