"""
策略迭代算法实现
"""
import numpy as np
import copy
from maze_env import MazeEnvironment

class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        
        self.states = env.get_all_states()
        self.n_states = len(self.states)
        
    
        self.V = {state: 0.0 for state in self.states}
        self.policy = {state: np.random.choice(env.actions) for state in self.states}
        
        self.value_history = []
        self.policy_history = []
        
    def policy_evaluation(self):
        while True:
            delta = 0
            V_old = copy.deepcopy(self.V)
            
            for state in self.states:
                if self.env.is_terminal(state):
                    continue
                
                # 计算当前状态的值
                action = self.policy[state]
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                
                # 更新值函数
                self.V[state] = reward + self.gamma * V_old[next_state]
                
                # 计算变化量
                delta = max(delta, abs(self.V[state] - V_old[state]))
            
            # 检查收敛
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """策略改进：基于当前值函数改进策略"""
        policy_stable = True
        
        for state in self.states:
            if self.env.is_terminal(state):
                continue
            
            old_action = self.policy[state]
            
            # 计算所有动作的Q值
            q_values = {}
            for action in self.env.actions:
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                q_values[action] = reward + self.gamma * self.V[next_state]
            
            # 选择最优动作
            best_action = max(q_values, key=q_values.get)
            self.policy[state] = best_action
            
            # 检查策略是否改变
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def solve(self, max_iterations=100):
        """执行策略迭代算法"""
        print("开始策略迭代...")
        
        for iteration in range(max_iterations):
            print(f"策略迭代第 {iteration + 1} 轮")
            
            # 策略评估
            self.policy_evaluation()
            
            # 保存当前值函数和策略
            self.value_history.append(copy.deepcopy(self.V))
            self.policy_history.append(copy.deepcopy(self.policy))
            
            # 策略改进
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                print(f"策略收敛，共迭代 {iteration + 1} 轮")
                break
        
        return self.V, self.policy
    
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
