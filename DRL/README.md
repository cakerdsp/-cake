# DQN算法解决CartPole倒立摆问题

本项目使用深度强化学习（Deep Reinforcement Learning）中的DQN（Deep Q-Network）算法来解决经典的CartPole倒立摆控制问题。

## 📋 项目简介

CartPole是强化学习领域的经典控制问题。本项目实现了标准的DQN算法，包含以下核心组件：

- **深度Q网络（Q-Network）**：使用PyTorch实现的三层全连接神经网络
- **经验回放（Experience Replay）**：存储和随机采样历史经验，提高训练稳定性
- **目标网络（Target Network）**：提供稳定的目标Q值，减少训练波动
- **Epsilon-Greedy策略**：平衡探索与利用

## 🔧 环境要求

- Python 3.7 - 3.12（推荐 3.8-3.11）
  - ⚠️ **注意**：如果使用conda安装，Python 3.12可能不兼容，建议使用pip或创建Python 3.10环境
- PyTorch 1.8.0 或更高版本
- Gymnasium（Gym的维护替代品，支持NumPy 2.0）
- NumPy
- Matplotlib

## 📦 依赖安装

### ⭐ 方法一：使用pip安装（强烈推荐，兼容性最好）

```bash
pip install torch gymnasium matplotlib numpy pygame
```

或者安装完整版本（包含所有依赖）：

```bash
pip install torch gymnasium[classic-control] matplotlib numpy
```

**优点**：pip通常支持最新的Python版本（包括3.12），安装简单快捷。使用Gymnasium替代已弃用的Gym，支持NumPy 2.0。pygame用于可视化渲染。

### 方法二：使用requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方法三：使用conda安装（如果遇到Python版本冲突）

如果当前环境是Python 3.12，conda-forge的gym可能不支持。有两种解决方案：

#### 方案A：创建新的conda环境（推荐）

```bash
# 创建Python 3.10的新环境
conda create -n rl_env python=3.10
conda activate rl_env

# 安装依赖
conda install pytorch -c pytorch
pip install gymnasium[classic-control] matplotlib numpy
```

#### 方案B：在当前环境使用pip安装

```bash
# 直接使用pip安装（pip支持Python 3.12）
pip install torch gymnasium[classic-control] matplotlib numpy
```

### 方法四：使用conda安装（Python 3.10及以下）

如果你的conda环境是Python 3.10或更低版本：

```bash
conda install pytorch -c pytorch
pip install gymnasium matplotlib numpy
```

**注意**：Gymnasium不在conda-forge中，建议使用pip安装。

## 🚀 快速开始

### 1. 运行训练程序

直接运行主程序即可开始训练：

```bash
python dqn_cartpole.py
```

### 2. 程序执行流程

程序将自动执行以下步骤：

1. **环境初始化**：创建CartPole-v1环境
2. **智能体初始化**：创建DQN智能体，初始化Q网络和目标网络
3. **训练阶段**：进行500轮训练，每轮最多500步
   - 每50轮会打印一次训练进度
   - 显示最近50轮的平均奖励和当前探索率
4. **结果可视化**：训练结束后自动生成并保存奖励曲线图 `result_plot.png`
5. **测试阶段**：使用训练好的模型进行10轮测试，显示测试结果
6. **倒立摆可视化**：自动生成并保存倒立摆的可视化图片 `cartpole_visualization.png`（包含6个关键帧）

### 3. 预期输出

训练过程中，你会看到类似以下的输出：

```
==================================================
DQN算法解决CartPole倒立摆问题
==================================================
状态维度: 4
动作维度: 2
==================================================
开始训练DQN...
训练轮数: 500
--------------------------------------------------
轮次 50/500 | 平均奖励（最近50轮）: 45.32 | 探索率: 0.779
轮次 100/500 | 平均奖励（最近50轮）: 78.56 | 探索率: 0.607
轮次 150/500 | 平均奖励（最近50轮）: 125.43 | 探索率: 0.473
轮次 200/500 | 平均奖励（最近50轮）: 165.78 | 探索率: 0.369
轮次 250/500 | 平均奖励（最近50轮）: 185.92 | 探索率: 0.288
轮次 300/500 | 平均奖励（最近50轮）: 195.34 | 探索率: 0.224
轮次 350/500 | 平均奖励（最近50轮）: 198.67 | 探索率: 0.175
轮次 400/500 | 平均奖励（最近50轮）: 199.45 | 探索率: 0.136
轮次 450/500 | 平均奖励（最近50轮）: 199.78 | 探索率: 0.106
轮次 500/500 | 平均奖励（最近50轮）: 199.89 | 探索率: 0.083
--------------------------------------------------
训练完成！

奖励曲线已保存至: result_plot.png

开始测试智能体...
测试轮次 1: 奖励 = 200
测试轮次 2: 奖励 = 200
...
平均测试奖励: 199.80
最高测试奖励: 200
最低测试奖励: 199

程序执行完成！
```

## 📁 项目文件说明

```
DRL/
├── dqn_cartpole.py              # 主程序文件，包含DQN算法实现
├── project_report.md            # 项目报告（学术报告）
├── README.md                    # 本文件
├── requirements.txt             # Python依赖包列表
├── result_plot.png              # 训练结果曲线图（运行后自动生成）
└── cartpole_visualization.png   # 倒立摆可视化图片（运行后自动生成）
```

### 文件详细说明

- **`dqn_cartpole.py`**：包含完整的DQN实现
  - `QNetwork`：Q网络类
  - `ReplayBuffer`：经验回放缓冲区类
  - `DQNAgent`：DQN智能体类
  - `train_dqn()`：训练函数
  - `plot_rewards()`：绘制奖励曲线
  - `test_agent()`：测试函数
  - `main()`：主函数

- **`project_report.md`**：详细的项目报告
  - 案例背景与任务定义
  - 算法核心原理（含数学公式）
  - 代码实现细节
  - 实验结果分析
  - 总结与展望

- **`result_plot.png`**：训练过程可视化结果
  - 蓝色曲线：每轮奖励（原始数据）
  - 红色曲线：移动平均（窗口=50）
  - 自动保存，无需手动操作

- **`cartpole_visualization.png`**：倒立摆可视化图片
  - 包含6个关键时间步的倒立摆状态
  - 展示训练好的智能体如何控制倒立摆保持平衡
  - 自动保存，无需手动操作

## ⚙️ 超参数配置

如果需要修改训练参数，可以在 `main()` 函数中调整 `DQNAgent` 的初始化参数：

```python
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=0.001,              # 学习率
    gamma=0.99,            # 折扣因子
    epsilon_start=1.0,     # 初始探索率
    epsilon_end=0.01,      # 最终探索率
    epsilon_decay=0.995,   # 探索率衰减系数
    memory_size=10000,     # 经验回放缓冲区大小
    batch_size=64,         # 批次大小
    target_update=10       # 目标网络更新频率
)
```

也可以修改训练轮数：

```python
episode_rewards = train_dqn(env, agent, num_episodes=500, max_steps=500)
```

## 📊 预期结果

### 训练曲线特征

1. **初期（0-100轮）**：奖励波动较大，平均奖励较低（50-100分）
2. **中期（100-300轮）**：奖励快速上升，学习效果显著
3. **后期（300-500轮）**：奖励稳定在接近200分，策略收敛

### 性能指标

- **最终平均奖励**：接近200分（CartPole-v1的最大奖励）
- **收敛速度**：约300-400轮达到稳定性能
- **测试成功率**：接近100%（能够稳定保持平衡）

## 🔍 常见问题

### Q1: 训练时间需要多久？

**A:** 在普通CPU上，500轮训练大约需要10-20分钟。如果使用GPU，训练速度会显著提升。

### Q2: 如何加快训练速度？

**A:** 可以：
- 减少训练轮数（如改为300轮）
- 使用GPU加速（需要安装支持CUDA的PyTorch）
- 减少经验回放缓冲区大小和批次大小

### Q3: 为什么奖励曲线波动很大？

**A:** 这是正常现象。DQN在训练初期会进行大量随机探索，导致奖励波动。随着训练进行，移动平均曲线会逐渐稳定上升。

### Q4: 如何保存训练好的模型？

**A:** 可以在代码中添加模型保存功能：

```python
# 训练结束后保存模型
torch.save(agent.q_network.state_dict(), 'dqn_model.pth')

# 加载模型
agent.q_network.load_state_dict(torch.load('dqn_model.pth'))
```

### Q5: 支持其他Gymnasium环境吗？

**A:** 可以，只需修改环境创建代码：

```python
env = gym.make('环境名称')  # 如 'MountainCar-v0', 'Acrobot-v1' 等
```

注意：不同环境的状态和动作空间可能不同，需要相应调整网络结构。代码使用Gymnasium（`import gymnasium as gym`），与Gym API兼容。

### Q6: conda安装gym时出现Python版本冲突怎么办？

**A:** 这是常见问题，因为conda-forge的gym包可能不支持Python 3.12。解决方案：

1. **推荐方案**：使用pip安装（pip支持更新的Python版本）
   ```bash
   pip install torch gymnasium[classic-control] matplotlib numpy
   ```

2. **或者**：创建Python 3.10的新conda环境
   ```bash
   conda create -n rl_env python=3.10
   conda activate rl_env
   pip install torch gymnasium[classic-control] matplotlib numpy
   ```

3. **检查Python版本**：
   ```bash
   python --version
   ```

### Q7: 出现"Gym has been unmaintained"或"numpy has no attribute 'bool8'"错误？

**A:** 这是因为旧版本的Gym不兼容NumPy 2.0。解决方案：

1. **卸载旧版Gym，安装Gymnasium**（推荐）：
   ```bash
   pip uninstall gym
   pip install gymnasium
   ```

2. **代码已更新**：代码已使用 `import gymnasium as gym`，与Gym API完全兼容。

### Q8: 如何查看倒立摆的可视化图片？

**A:** 程序训练完成后会自动生成 `cartpole_visualization.png` 文件，包含6个关键时间步的倒立摆状态图片。如果训练过程中没有生成，可以：

1. **确保已安装pygame**：
   ```bash
   pip install pygame
   # 或
   pip install gymnasium[classic-control]
   ```

2. **重新运行程序**：程序会在训练和测试后自动生成可视化图片

3. **手动调用可视化函数**（在代码中）：
   ```python
   visualize_agent(agent, num_episodes=1, save_images=True, 
                   save_path='cartpole_visualization.png')
   ```

**注意**：如果出现"pygame is not installed"错误，请安装pygame依赖。

### Q9: 运行时出现"OMP: Error #15"或"libiomp5md.dll already initialized"错误？

**A:** 这是Windows下常见的OpenMP库冲突问题（PyTorch/NumPy/MKL库冲突）。代码中已经自动处理了这个问题，如果仍然出现错误，可以：

1. **代码已自动处理**：程序开头已设置 `KMP_DUPLICATE_LIB_OK=TRUE`，通常可以解决

2. **手动设置环境变量**（如果仍有问题）：
   ```powershell
   # PowerShell
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python dqn_cartpole.py
   ```
   
   ```cmd
   # CMD
   set KMP_DUPLICATE_LIB_OK=TRUE
   python dqn_cartpole.py
   ```

3. **永久设置**（可选）：在系统环境变量中添加 `KMP_DUPLICATE_LIB_OK=TRUE`

## 📚 参考资料

- [DQN原始论文](https://www.nature.com/articles/nature14236)
- [OpenAI Gym文档](https://www.gymlibrary.dev/)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

## 📝 许可证

本项目仅用于学习和研究目的。

## 👤 作者

深度强化学习课程作业

---

**提示**：首次运行前请确保已安装所有依赖。如果遇到任何问题，请检查Python版本和依赖包的兼容性。

