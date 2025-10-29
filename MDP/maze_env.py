"""
迷宫环境类 - 实现8x8迷宫环境
使用二维数组存储地图，便于自定义修改
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MazeEnvironment:
    """8x8迷宫环境类"""
    
    def __init__(self, custom_map=None):
        """
        初始化迷宫环境
        
        Args:
            custom_map: 自定义地图，如果为None则使用默认地图
                       地图格式：0=路径，1=墙壁，2=起始点，3=目标点
        """
        self.size = 8
    
        self.default_map = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],  # 第0行：起始点在(0,0)
            [1, 0, 0, 0, 0, 0, 0, 1],  # 第1行：墙壁在(1,1), (1,3), (1,5), (1,7)
            [2, 0, 1, 1, 0, 1, 0, 1],  # 第2行：墙壁在(2,1), (2,3), (2,5)
            [1, 0, 0, 1, 1, 0, 0, 1],  # 第3行：墙壁在(3,1), (3,3), (3,5), (3,6)
            [1, 1, 0, 0, 1, 0, 1, 1],  # 第4行：墙壁在(4,1), (4,3)
            [1, 0, 1, 0, 1, 0, 0, 1],  # 第5行：墙壁在(5,1), (5,3), (5,4), (5,5), (5,6)
            [1, 0, 0, 0, 0, 1, 0, 3],  # 第6行：墙壁在(6,1)
            [1, 1, 1, 1, 1, 1, 1, 1]   # 第7行：墙壁在(7,3), (7,5)，目标点在(7,7)
        ])
        
        # 使用自定义地图或默认地图
        if custom_map is not None:
            self.map = np.array(custom_map)
            if self.map.shape != (self.size, self.size):
                raise ValueError(f"地图大小必须是 {self.size}x{self.size}")
        else:
            self.map = self.default_map.copy()
        
        # 从地图中提取关键位置
        self.start = self._find_position(2)  # 起始位置
        self.goal = self._find_position(3)   # 目标位置
        self.walls = self._find_walls()       # 墙壁位置列表
        
        # 动作定义：北、东、南、西
        self.actions = ['N', 'E', 'S', 'W']
        self.action_to_delta = {
            'N': (-1, 0),
            'E': (0, 1),
            'S': (1, 0),
            'W': (0, -1)
        }
        
        # 奖励设置
        self.step_reward = -1
        self.goal_reward = 0
    
    def _find_position(self, value):
        """在地图中查找指定值的位置"""
        positions = np.where(self.map == value)
        if len(positions[0]) == 0:
            raise ValueError(f"地图中没有找到值为 {value} 的位置")
        if len(positions[0]) > 1:
            raise ValueError(f"地图中有多个值为 {value} 的位置")
        return (positions[0][0], positions[1][0])
    
    def _find_walls(self):
        """在地图中查找所有墙壁位置"""
        wall_positions = np.where(self.map == 1)
        return list(zip(wall_positions[0], wall_positions[1]))
    
    def set_custom_map(self, custom_map):
        """设置自定义地图"""
        if custom_map.shape != (self.size, self.size):
            raise ValueError(f"地图大小必须是 {self.size}x{self.size}")
        
        # 检查地图是否包含起始点和目标点
        if not np.any(custom_map == 2):
            raise ValueError("地图必须包含起始点（值为2）")
        if not np.any(custom_map == 3):
            raise ValueError("地图必须包含目标点（值为3）")
        
        self.map = custom_map.copy()
        self.start = self._find_position(2)
        self.goal = self._find_position(3)
        self.walls = self._find_walls()
    
    def get_map(self):
        """获取当前地图"""
        return self.map.copy()
    
    def print_map(self):
        """打印地图（用于调试）"""
        print("当前地图:")
        print("0=路径, 1=墙壁, 2=起始点, 3=目标点")
        print(self.map)
        print(f"起始位置: {self.start}")
        print(f"目标位置: {self.goal}")
        print(f"墙壁数量: {len(self.walls)}")
        
    def is_valid_position(self, pos):
        """检查位置是否有效（不在墙壁内且在边界内）"""
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos in self.walls:
            return False
        return True
    
    def get_next_state(self, state, action):
        """根据当前状态和动作获取下一个状态"""
        row, col = state
        delta_row, delta_col = self.action_to_delta[action]
        next_row, next_col = row + delta_row, col + delta_col
        
        # 如果下一个位置无效，则保持在当前位置
        if not self.is_valid_position((next_row, next_col)):
            return state
        
        return (next_row, next_col)
    
    def get_reward(self, state, action, next_state):
        """获取奖励"""
        if next_state == self.goal:
            return self.goal_reward
        return self.step_reward
    
    def is_terminal(self, state):
        """检查是否为终止状态"""
        return state == self.goal
    
    def get_all_states(self):
        """获取所有有效状态"""
        states = []
        for row in range(self.size):
            for col in range(self.size):
                if (row, col) not in self.walls:
                    states.append((row, col))
        return states
    
    def visualize_maze(self, title="迷宫环境"):
        """可视化迷宫"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # 绘制墙壁（黑色）
        for wall in self.walls:
            rect = patches.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1, 
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        # 绘制起始位置（绿色）
        start_rect = patches.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 1, 1,
                                     linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(start_rect)
        ax.text(self.start[1], self.start[0], 'Start', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制目标位置（红色）
        goal_rect = patches.Rectangle((self.goal[1] - 0.5, self.goal[0] - 0.5), 1, 1,
                                    linewidth=2, edgecolor='red', facecolor='lightcoral')
        ax.add_patch(goal_rect)
        ax.text(self.goal[1], self.goal[0], 'Goal', ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        # 设置坐标轴
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.invert_yaxis()  # 让(0,0)在左上角
        
        plt.tight_layout()
        return fig, ax
