"""
验证CartPole环境参数 - 修正版
真正从环境内核读取阈值，并正确注入状态进行测试
"""

import gymnasium as gym
import numpy as np

def verify_cartpole_params_fixed():
    print("=" * 60)
    print("CartPole-v1 环境参数深度验证 (修正版)")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    
    # --- 关键步骤：先获取真值 (Ground Truth) ---
    # 使用 unwrapped 访问环境核心逻辑，这是“不做假”的关键
    core_env = env.unwrapped
    
    # 动态获取真实阈值，而不是写死 2.4 或 12度
    REAL_X_THRESHOLD = getattr(core_env, 'x_threshold', 2.4) # 默认值作为fallback
    REAL_THETA_THRESHOLD = getattr(core_env, 'theta_threshold_radians', 0.2095)
    MAX_STEPS = env.spec.max_episode_steps if env.spec else 500
    
    print("\n【1. 读取环境核心参数 (Ground Truth)】")
    print(f"  真实 X 轴阈值: {REAL_X_THRESHOLD} 米")
    print(f"  真实角度阈值 : {REAL_THETA_THRESHOLD} 弧度 ({np.degrees(REAL_THETA_THRESHOLD):.2f}°)")
    print(f"  最大步数限制 : {MAX_STEPS}")

    # --- 验证 1: 物理边界测试 ---
    print("\n【2. 物理边界注入测试】")
    # 我们将小车强制放到临界点，看一步之后是否终止
    
    # 测试案例 A: 设置在边界内一点点 (应该存活)
    env.reset()
    # 强制修改环境内部状态: [x, x_dot, theta, theta_dot]
    # 注意：在Gym中，直接修改 env.unwrapped.state
    safe_x = REAL_X_THRESHOLD - 0.05
    core_env.state = np.array([safe_x, 0.01, 0, 0]) 
    _, _, term, trunc, _ = env.step(1) # 向右推
    print(f"  测试A (边界内 {safe_x:.2f}m): 终止状态={term or trunc} (预期: False)")

    # 测试案例 B: 设置在边界外 (应该立即终止)
    env.reset()
    danger_x = REAL_X_THRESHOLD + 0.05
    core_env.state = np.array([danger_x, 0.01, 0, 0])
    _, _, term, trunc, _ = env.step(1)
    print(f"  测试B (边界外 {danger_x:.2f}m): 终止状态={term} (预期: True)")
    
    # 测试案例 C: 角度测试
    env.reset()
    danger_theta = REAL_THETA_THRESHOLD + 0.01
    core_env.state = np.array([0, 0, danger_theta, 0])
    _, _, term, trunc, _ = env.step(0)
    print(f"  测试C (角度超限 {np.degrees(danger_theta):.2f}°): 终止状态={term} (预期: True)")

    # --- 验证 2: 运行循环中的动态判断 ---
    print("\n【3. 运行循环验证 (不使用硬编码数字)】")
    
    state, _ = env.reset()
    reasons = []
    
    for step in range(MAX_STEPS + 10): # 多跑几步以测试截断
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            # 使用前面读取的 REAL_X_THRESHOLD 变量，而不是写死 2.4
            current_x = next_state[0]
            current_theta = next_state[2]
            
            if abs(current_x) >= REAL_X_THRESHOLD:
                reasons.append(f"位置越界 (x={current_x:.4f} >= {REAL_X_THRESHOLD})")
            elif abs(current_theta) >= REAL_THETA_THRESHOLD:
                reasons.append(f"角度越界 (θ={current_theta:.4f} >= {REAL_THETA_THRESHOLD})")
            elif truncated: # 通常对应 TimeLimit
                reasons.append(f"达到最大步数 (step={step+1})")
            else:
                reasons.append("未知原因")
            
            print(f"  Episode 结束于第 {step+1} 步. 原因: {reasons[-1]}")
            break
            
    env.close()
    print("\n验证完成：所有判断标准均来自环境内部属性。")

if __name__ == '__main__':
    verify_cartpole_params_fixed()