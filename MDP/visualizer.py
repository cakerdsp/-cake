"""
可视化功能模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, env):
        self.env = env
        self.size = env.size
        
    def plot_value_function(self, V, title="值函数", save_path=None):
        """绘制值函数热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建值函数矩阵
        value_matrix = np.full((self.size, self.size), np.nan)
        for state, value in V.items():
            row, col = state
            value_matrix[row, col] = value
        
        # 绘制热力图 - 注意：不使用invert_yaxis，让(0,0)在左上角
        im = ax.imshow(value_matrix, cmap='RdYlBu_r', aspect='equal', origin='upper')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('状态值', fontsize=12)
        
        # 绘制网格
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels(range(self.size))
        ax.set_yticklabels(range(self.size))
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=2, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 标记起始和目标位置
        ax.add_patch(patches.Rectangle((self.env.start[1] - 0.5, self.env.start[0] - 0.5), 
                                     1, 1, linewidth=3, edgecolor='green', facecolor='none'))
        ax.text(self.env.start[1], self.env.start[0], 'S', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='green')
        
        ax.add_patch(patches.Rectangle((self.env.goal[1] - 0.5, self.env.goal[0] - 0.5), 
                                     1, 1, linewidth=3, edgecolor='red', facecolor='none'))
        ax.text(self.env.goal[1], self.env.goal[0], 'G', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='red')
        
        # 在非墙壁位置显示值
        for state, value in V.items():
            if not self.env.is_terminal(state):
                ax.text(state[1], state[0], f'{value:.2f}', ha='center', va='center', 
                       fontsize=8, color='white' if value < -5 else 'black')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_policy(self, policy, title="最优策略", save_path=None):
        """绘制策略箭头图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 绘制策略箭头
        arrow_map = {
            'N': (0, -0.3),  # 北：向上，在matplotlib中y轴倒置，所以是负值
            'E': (0.3, 0),   # 东：向右
            'S': (0, 0.3),   # 南：向下，在matplotlib中y轴倒置，所以是正值
            'W': (-0.3, 0)   # 西：向左
        }
        
        for state, action in policy.items():
            if action is None or self.env.is_terminal(state):
                continue
            
            row, col = state
            dx, dy = arrow_map[action]
            
            ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue', linewidth=2)
        
        # 标记起始和目标位置
        ax.add_patch(patches.Rectangle((self.env.start[1] - 0.5, self.env.start[0] - 0.5), 
                                     1, 1, linewidth=3, edgecolor='green', facecolor='lightgreen'))
        ax.text(self.env.start[1], self.env.start[0], 'Start', ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        ax.add_patch(patches.Rectangle((self.env.goal[1] - 0.5, self.env.goal[0] - 0.5), 
                                     1, 1, linewidth=3, edgecolor='red', facecolor='lightcoral'))
        ax.text(self.env.goal[1], self.env.goal[0], 'Goal', ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        # 设置坐标轴
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.invert_yaxis()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_path(self, path, title="最优路径", save_path=None):
        """绘制最优路径"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 绘制路径
        if len(path) > 1:
            path_x = [pos[1] for pos in path]
            path_y = [pos[0] for pos in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7)
            
            # 标记路径点
            for i, (row, col) in enumerate(path):
                if i == 0:  # 起始点
                    ax.plot(col, row, 'go', markersize=12, markeredgecolor='darkgreen')
                    ax.text(col, row, 'S', ha='center', va='center', fontsize=10, fontweight='bold')
                elif i == len(path) - 1:  # 目标点
                    ax.plot(col, row, 'ro', markersize=12, markeredgecolor='darkred')
                    ax.text(col, row, 'G', ha='center', va='center', fontsize=10, fontweight='bold')
                else:  # 中间点
                    ax.plot(col, row, 'bo', markersize=8, alpha=0.7)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        # 设置坐标轴
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.invert_yaxis()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_convergence(self, value_history, title="值函数收敛过程", save_path=None):
        """绘制收敛过程"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算每个状态的值变化
        iterations = range(len(value_history))
        
        # 选择几个代表性状态进行绘制
        representative_states = [
            self.env.start,
            (0, 1), (1, 0), (2, 0), (3, 0),
            (0, 2), (1, 2), (2, 2), (3, 2)
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(representative_states)))
        
        for i, state in enumerate(representative_states):
            if state in value_history[0]:  # 确保状态存在
                values = [vh[state] for vh in value_history]
                ax.plot(iterations, values, color=colors[i], linewidth=2, 
                       label=f'状态{state}', marker='o', markersize=4)
        
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('状态值', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_comparison(self, pi_results, vi_results, save_path=None):
        """比较策略迭代和值迭代的结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 策略迭代结果
        pi_V, pi_policy = pi_results
        pi_path, pi_reward = self._simulate_path(pi_policy)
        
        # 值迭代结果
        vi_V, vi_policy = vi_results
        vi_path, vi_reward = self._simulate_path(vi_policy)
        
        # 第一行：策略迭代
        self._plot_value_function_subplot(axes[0, 0], pi_V, "策略迭代 - 值函数")
        self._plot_policy_subplot(axes[0, 1], pi_policy, "策略迭代 - 策略")
        self._plot_path_subplot(axes[0, 2], pi_path, f"策略迭代 - 路径 (奖励: {pi_reward:.2f})")
        
        # 第二行：值迭代
        self._plot_value_function_subplot(axes[1, 0], vi_V, "值迭代 - 值函数")
        self._plot_policy_subplot(axes[1, 1], vi_policy, "值迭代 - 策略")
        self._plot_path_subplot(axes[1, 2], vi_path, f"值迭代 - 路径 (奖励: {vi_reward:.2f})")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def _simulate_path(self, policy):
        """模拟路径"""
        path = [self.env.start]
        current_state = self.env.start
        total_reward = 0
        
        for step in range(100):
            if self.env.is_terminal(current_state):
                break
            
            action = policy[current_state]
            next_state = self.env.get_next_state(current_state, action)
            reward = self.env.get_reward(current_state, action, next_state)
            
            path.append(next_state)
            total_reward += reward
            current_state = next_state
        
        return path, total_reward
    
    def _plot_value_function_subplot(self, ax, V, title):
        """在子图中绘制值函数"""
        value_matrix = np.full((self.size, self.size), np.nan)
        for state, value in V.items():
            row, col = state
            value_matrix[row, col] = value
        
        im = ax.imshow(value_matrix, cmap='RdYlBu_r', aspect='equal', origin='upper')
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=2, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
    
    def _plot_policy_subplot(self, ax, policy, title):
        """在子图中绘制策略"""
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 绘制策略箭头
        arrow_map = {
            'N': (0, -0.2),  # 北：向上，在matplotlib中y轴倒置，所以是负值
            'E': (0.2, 0),   # 东：向右
            'S': (0, 0.2),   # 南：向下，在matplotlib中y轴倒置，所以是正值
            'W': (-0.2, 0)   # 西：向左
        }
        
        for state, action in policy.items():
            if action is None or self.env.is_terminal(state):
                continue
            
            row, col = state
            dx, dy = arrow_map[action]
            
            ax.arrow(col, row, dx, dy, head_width=0.05, head_length=0.05, 
                    fc='blue', ec='blue', linewidth=1)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.invert_yaxis()
    
    def _plot_path_subplot(self, ax, path, title):
        """在子图中绘制路径"""
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # 绘制墙壁
        for wall in self.env.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 绘制路径
        if len(path) > 1:
            path_x = [pos[1] for pos in path]
            path_y = [pos[0] for pos in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
            
            # 标记路径点
            for i, (row, col) in enumerate(path):
                if i == 0:  # 起始点
                    ax.plot(col, row, 'go', markersize=8)
                elif i == len(path) - 1:  # 目标点
                    ax.plot(col, row, 'ro', markersize=8)
                else:  # 中间点
                    ax.plot(col, row, 'bo', markersize=4, alpha=0.7)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.invert_yaxis()
