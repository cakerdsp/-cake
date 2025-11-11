"""
Cliff Walk 环境实现。

环境设置与 Sutton & Barto 经典教材中的 Cliff Walking 任务保持一致：
- 网格大小为 4 行 × 12 列。
- 起点位于左下角 (3, 0)，终点位于右下角 (3, 11)。
- 除起点与终点之外，下方一整行（列索引 1~10）均为悬崖。
- 每一步奖励为 -1，跌落悬崖时奖励 -100 并立即回到起点继续。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Action = int  # 0: 向上, 1: 向右, 2: 向下, 3: 向左
State = Tuple[int, int]


@dataclass(frozen=True)
class Transition:
    """单步转移信息。"""

    next_state: State
    reward: float
    terminated: bool


class CliffWalkEnv:
    """Cliff Walk 环境。"""

    ACTIONS: Dict[Action, Tuple[int, int]] = {
        0: (-1, 0),  # up
        1: (0, 1),  # right
        2: (1, 0),  # down
        3: (0, -1),  # left
    }

    def __init__(self) -> None:
        self._n_rows = 4
        self._n_cols = 12
        self._start_state: State = (self._n_rows - 1, 0)
        self._terminal_state: State = (self._n_rows - 1, self._n_cols - 1)
        self._cliff_states = {
            (self._n_rows - 1, c) for c in range(1, self._n_cols - 1)
        }
        self._state: State = self._start_state

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    @property
    def n_states(self) -> int:
        return self._n_rows * self._n_cols

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def start_state(self) -> State:
        return self._start_state

    @property
    def terminal_state(self) -> State:
        return self._terminal_state

    @property
    def cliff_states(self) -> set[State]:
        return set(self._cliff_states)

    def reset(self) -> State:
        """重置环境，返回起始状态。"""
        self._state = self._start_state
        return self._state

    def step(self, action: Action) -> Transition:
        """执行动作，返回转移信息。"""
        if action not in self.ACTIONS:
            raise ValueError(f"非法动作 {action}，应在 0~{len(self.ACTIONS) - 1} 之间。")

        if self._state == self._terminal_state:
            # 已经在终点，则保持不动
            return Transition(self._state, 0.0, True)

        move = self.ACTIONS[action]
        next_state = self._move(self._state, move)

        if next_state in self._cliff_states:
            reward = -100.0
            self._state = self._start_state
            terminated = False
        elif next_state == self._terminal_state:
            reward = -1.0
            self._state = next_state
            terminated = True
        else:
            reward = -1.0
            self._state = next_state
            terminated = False

        return Transition(self._state, reward, terminated)

    def _move(self, state: State, delta: Tuple[int, int]) -> State:
        r, c = state
        dr, dc = delta
        nr = min(max(r + dr, 0), self._n_rows - 1)
        nc = min(max(c + dc, 0), self._n_cols - 1)
        return nr, nc

    def state_to_index(self, state: State) -> int:
        """将二维坐标状态映射为一维索引，便于 Q 表表示。"""
        r, c = state
        return r * self._n_cols + c

    def index_to_state(self, index: int) -> State:
        """将一维索引映射回二维坐标状态。"""
        if index < 0 or index >= self.n_states:
            raise ValueError(f"索引 {index} 越界。")
        r, c = divmod(index, self._n_cols)
        return r, c

    def legal_states(self) -> List[State]:
        """列出所有可访问状态（不含悬崖）。"""
        states = []
        for r in range(self._n_rows):
            for c in range(self._n_cols):
                state = (r, c)
                if state in self._cliff_states:
                    continue
                states.append(state)
        return states

    def render_ascii(self, agent_state: State | None = None) -> str:
        """返回 ASCII 形式的网格表示，用于调试。"""
        rows: List[str] = []
        for r in range(self._n_rows):
            cells = []
            for c in range(self._n_cols):
                state = (r, c)
                if state == agent_state:
                    cells.append("A")
                elif state == self._start_state:
                    cells.append("S")
                elif state == self._terminal_state:
                    cells.append("G")
                elif state in self._cliff_states:
                    cells.append("X")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)


def seed_everything(seed: int) -> None:
    """设置随机种子，确保可复现性。"""
    np.random.seed(seed)


