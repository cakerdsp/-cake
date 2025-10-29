"""
值迭代算法实现
"""
import numpy as np
import copy
from maze_env import MazeEnvironment

class ValueIteration:
    """值迭代算法"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        初始化值迭代算法
        
        Args:
            env: 迷宫环境
            gamma: 折扣因子
            theta: 收敛阈值
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # 获取所有状态
        self.states = env.get_all_states()
        self.n_states = len(self.states)
        
        # 初始化值函数
        self.V = {state: 0.0 for state in self.states}
        
        # 存储迭代历史用于可视化
        self.value_history = []
        
    def value_iteration_step(self):
        """执行一步值迭代"""
        V_old = copy.deepcopy(self.V)
        delta = 0
        
        for state in self.states:
            if self.env.is_terminal(state):
                continue
            
            # 计算所有动作的Q值
            q_values = []
            for action in self.env.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                q_value = reward + self.gamma * V_old[next_state]
                q_values.append(q_value)
            
            # 更新值函数（取最大值）
            self.V[state] = max(q_values)
            
            # 计算变化量
            delta = max(delta, abs(self.V[state] - V_old[state]))
        
        return delta
    
    def solve(self, max_iterations=1000):
        """执行值迭代算法"""
        print("开始值迭代...")
        
        for iteration in range(max_iterations):
            # 保存当前值函数
            self.value_history.append(copy.deepcopy(self.V))
            
            # 执行一步值迭代
            delta = self.value_iteration_step()
            
            if iteration % 50 == 0:
                print(f"值迭代第 {iteration + 1} 轮，变化量: {delta:.6f}")
            
            # 检查收敛
            if delta < self.theta:
                print(f"值函数收敛，共迭代 {iteration + 1} 轮")
                break
        
        # 基于收敛的值函数提取最优策略
        self.policy = self.extract_policy()
        
        return self.V, self.policy
    
    def extract_policy(self):
        """从值函数中提取最优策略"""
        policy = {}
        
        for state in self.states:
            if self.env.is_terminal(state):
                policy[state] = None
                continue
            
            # 计算所有动作的Q值
            q_values = {}
            for action in self.env.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                q_values[action] = reward + self.gamma * self.V[next_state]
            
            # 选择最优动作
            best_action = max(q_values, key=q_values.get)
            policy[state] = best_action
        
        return policy
    
    def get_q_values(self, state):
        """获取指定状态的Q值"""
        q_values = {}
        for action in self.env.actions:
            next_state = self.env.get_next_state(state, action)
            reward = self.env.get_reward(state, action, next_state)
            q_values[action] = reward + self.gamma * self.V[next_state]
        return q_values
    
    def get_policy_arrow(self, state):
        """获取策略箭头方向"""
        if self.env.is_terminal(state):
            return '★'
        
        action = self.policy[state]
        arrow_map = {
            'N': '↑',
            'E': '→',
            'S': '↓',
            'W': '←'
        }
        return arrow_map[action]
    
    def simulate_path(self, max_steps=100):
        """模拟最优路径"""
        path = [self.env.start]
        current_state = self.env.start
        total_reward = 0
        
        for step in range(max_steps):
            if self.env.is_terminal(current_state):
                break
            
            action = self.policy[current_state]
            next_state = self.env.get_next_state(current_state, action)
            reward = self.env.get_reward(current_state, action, next_state)
            
            path.append(next_state)
            total_reward += reward
            current_state = next_state
        
        return path, total_reward
