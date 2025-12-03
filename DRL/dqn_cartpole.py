import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)
        
        current_q_values = self.q_network(state).gather(1, action)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
            target_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 【修正】删除了这里的epsilon衰减，移到了每轮结束时
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, agent, num_episodes=500, max_steps=500):
    episode_rewards = []
    best_avg_reward = 0  # 用于记录历史最高平均分
    save_path = 'best_cartpole_model.pth'
    
    print("开始训练DQN (含自动保存最佳模型)...")
    print(f"训练轮数: {num_episodes}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            real_done = terminated
            
            agent.remember(state, action, reward, next_state, real_done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # 【修正】每轮结束衰减一次探索率（防止过早停止探索）
        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay
            
        episode_rewards.append(total_reward)
        
        # === 核心功能：保存最佳模型 ===
        # 计算最近50轮的平均分
        if len(episode_rewards) >= 50:
            avg_reward = np.mean(episode_rewards[-50:])
            # 如果创新高，就保存
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.q_network.state_dict(), save_path)
                print(f"  >>> 新纪录！平均分: {avg_reward:.2f}，模型已保存。")
        
        # 打印日志
        if (episode + 1) % 50 == 0:
            current_avg = np.mean(episode_rewards[-50:])
            print(f"轮次 {episode + 1}/{num_episodes} | "
                  f"平均奖励: {current_avg:.2f} | "
                  f"最佳平均: {best_avg_reward:.2f} | "
                  f"探索率: {agent.epsilon:.3f}")
    
    print("-" * 50)
    print(f"训练完成！最佳模型平均分: {best_avg_reward:.2f}")
    return episode_rewards

def plot_rewards(episode_rewards, save_path='result_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='每轮奖励')
    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = [np.mean(episode_rewards[max(0, i - window_size + 1):i+1]) for i in range(len(episode_rewards))]
        plt.plot(moving_avg, color='red', linewidth=2, label=f'移动平均（窗口={window_size}）')
    
    plt.xlabel('训练轮次')
    plt.ylabel('奖励')
    plt.title('DQN训练曲线 (含最佳模型保存)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n奖励曲线已保存至: {save_path}")

def visualize_best_model(agent, save_path='cartpole_best_vis.png'):
    """加载保存的最佳模型并可视化"""
    model_path = 'best_cartpole_model.pth'
    if not os.path.exists(model_path):
        print("未找到最佳模型文件，跳过可视化。")
        return

    # 加载权重
    print(f"\n加载最佳模型权重: {model_path} ...")
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval() # 切换到评估模式

    try:
        import pygame
        env = gym.make('CartPole-v1', render_mode='rgb_array')
    except:
        print("无法创建渲染环境，跳过。")
        return

    frames = []
    state, _ = env.reset()
    total_reward = 0
    
    # 跑一局演示
    for _ in range(500):
        frame = env.render()
        if frame is not None: frames.append(frame)
        
        action = agent.select_action(state, training=False)
        state, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: 
            # 补最后一帧
            f = env.render()
            if f is not None: frames.append(f)
            break
            
    env.close()
    print(f"最佳模型演示得分: {total_reward}")
    
    # 拼图保存
    if len(frames) > 0:
        num_key = 6
        idx_list = np.linspace(0, len(frames)-1, num_key, dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'最佳DQN模型演示 (得分: {total_reward})', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(idx_list):
                ax.imshow(frames[idx_list[idx]])
                ax.set_title(f'Frame {idx_list[idx]}')
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"演示图片已保存至: {save_path}")

def test_best_model(agent, num_episodes=10):
    """测试最佳模型的平均性能"""
    model_path = 'best_cartpole_model.pth'
    if not os.path.exists(model_path):
        return

    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()
    
    env = gym.make('CartPole-v1')
    rewards = []
    print("\n=== 测试最佳模型 (10轮) ===")
    
    for i in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state, training=False)
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc: break
        rewards.append(ep_reward)
        print(f"测试轮次 {i+1}: {ep_reward}")
        
    print(f"平均分: {np.mean(rewards):.2f}")
    env.close()

def main():
    env = gym.make('CartPole-v1', max_episode_steps=500)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0005,
        target_update=200
    )
    
    # 1. 训练并自动保存最佳模型
    episode_rewards = train_dqn(env, agent, num_episodes=500)
    
    # 2. 绘制训练曲线
    plot_rewards(episode_rewards)
    
    # 3. 加载刚才保存的最佳模型进行测试
    test_best_model(agent)
    
    # 4. 可视化最佳模型
    visualize_best_model(agent)
    
    print("\n程序执行完成！")

if __name__ == '__main__':
    main()