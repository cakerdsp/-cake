"""
主程序 - 运行策略迭代和值迭代算法并比较结果
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from maze_env import MazeEnvironment
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from visualizer import Visualizer

def main():
    """主函数"""
    print("=" * 60)
    print("强化学习迷宫问题求解")
    print("策略迭代 vs 值迭代")
    print("=" * 60)
    
    # 创建迷宫环境（使用默认地图）
    env = MazeEnvironment()
    visualizer = Visualizer(env)
    
    # 显示迷宫环境
    print("\n1. 迷宫环境")
    print(f"迷宫大小: {env.size}x{env.size}")
    print(f"起始位置: {env.start}")
    print(f"目标位置: {env.goal}")
    print(f"墙壁数量: {len(env.walls)}")
    print(f"有效状态数: {len(env.get_all_states())}")
    
    # 显示地图
    env.print_map()
    
    # 可视化迷宫
    fig, ax = env.visualize_maze("8x8迷宫环境")
    plt.show()
    
    # 算法参数
    gamma = 0.9
    theta = 1e-6
    
    print(f"\n算法参数:")
    print(f"折扣因子 γ: {gamma}")
    print(f"收敛阈值 θ: {theta}")
    
    # 运行策略迭代
    print("\n" + "=" * 40)
    print("2. 策略迭代算法")
    print("=" * 40)
    
    pi_solver = PolicyIteration(env, gamma=gamma, theta=theta)
    start_time = time.time()
    pi_V, pi_policy = pi_solver.solve()
    pi_time = time.time() - start_time
    
    print(f"策略迭代完成，用时: {pi_time:.4f}秒")
    print(f"迭代轮数: {len(pi_solver.value_history)}")
    
    # 运行值迭代
    print("\n" + "=" * 40)
    print("3. 值迭代算法")
    print("=" * 40)
    
    vi_solver = ValueIteration(env, gamma=gamma, theta=theta)
    start_time = time.time()
    vi_V, vi_policy = vi_solver.solve()
    vi_time = time.time() - start_time
    
    print(f"值迭代完成，用时: {vi_time:.4f}秒")
    print(f"迭代轮数: {len(vi_solver.value_history)}")
    
    # 比较结果
    print("\n" + "=" * 40)
    print("4. 结果比较")
    print("=" * 40)
    
    # 模拟最优路径
    pi_path, pi_reward = pi_solver.simulate_path()
    vi_path, vi_reward = vi_solver.simulate_path()
    
    print(f"策略迭代:")
    print(f"  - 路径长度: {len(pi_path)}")
    print(f"  - 总奖励: {pi_reward:.2f}")
    print(f"  - 计算时间: {pi_time:.4f}秒")
    print(f"  - 迭代次数: {len(pi_solver.value_history)}")
    
    print(f"\n值迭代:")
    print(f"  - 路径长度: {len(vi_path)}")
    print(f"  - 总奖励: {vi_reward:.2f}")
    print(f"  - 计算时间: {vi_time:.4f}秒")
    print(f"  - 迭代次数: {len(vi_solver.value_history)}")
    
    # 比较值函数
    value_diff = 0
    for state in env.get_all_states():
        if not env.is_terminal(state):
            diff = abs(pi_V[state] - vi_V[state])
            value_diff = max(value_diff, diff)
    
    print(f"\n值函数最大差异: {value_diff:.6f}")
    
    # 比较策略
    policy_diff = 0
    for state in env.get_all_states():
        if not env.is_terminal(state):
            if pi_policy[state] != vi_policy[state]:
                policy_diff += 1
    
    print(f"策略差异状态数: {policy_diff}")
    
    # 可视化结果
    print("\n" + "=" * 40)
    print("5. 结果可视化")
    print("=" * 40)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 策略迭代结果可视化
    print("生成策略迭代结果图...")
    visualizer.plot_value_function(pi_V, "策略迭代 - 值函数", "results/pi_value_function.png")
    visualizer.plot_policy(pi_policy, "策略迭代 - 最优策略", "results/pi_policy.png")
    visualizer.plot_path(pi_path, "策略迭代 - 最优路径", "results/pi_path.png")
    visualizer.plot_convergence(pi_solver.value_history, "策略迭代 - 收敛过程", "results/pi_convergence.png")
    
    # 值迭代结果可视化
    print("生成值迭代结果图...")
    visualizer.plot_value_function(vi_V, "值迭代 - 值函数", "results/vi_value_function.png")
    visualizer.plot_policy(vi_policy, "值迭代 - 最优策略", "results/vi_policy.png")
    visualizer.plot_path(vi_path, "值迭代 - 最优路径", "results/vi_path.png")
    visualizer.plot_convergence(vi_solver.value_history, "值迭代 - 收敛过程", "results/vi_convergence.png")
    
    # 比较图
    print("生成算法比较图...")
    visualizer.plot_comparison((pi_V, pi_policy), (vi_V, vi_policy), "results/comparison.png")
    
    # 显示所有图
    plt.show()
    
    # 详细分析
    print("\n" + "=" * 40)
    print("6. 详细分析")
    print("=" * 40)
    
    print("策略迭代分析:")
    print(f"  - 平均每轮用时: {pi_time/len(pi_solver.value_history):.4f}秒")
    print(f"  - 起始状态值: {pi_V[env.start]:.4f}")
    print(f"  - 目标状态值: {pi_V[env.goal]:.4f}")
    
    print("\n值迭代分析:")
    print(f"  - 平均每轮用时: {vi_time/len(vi_solver.value_history):.4f}秒")
    print(f"  - 起始状态值: {vi_V[env.start]:.4f}")
    print(f"  - 目标状态值: {vi_V[env.goal]:.4f}")
    
    # 显示最优路径
    print(f"\n策略迭代最优路径:")
    for i, state in enumerate(pi_path):
        print(f"  步骤 {i}: {state}")
    
    print(f"\n值迭代最优路径:")
    for i, state in enumerate(vi_path):
        print(f"  步骤 {i}: {state}")
    
    # 保存结果到文件
    print("\n" + "=" * 40)
    print("7. 保存结果")
    print("=" * 40)
    
    save_results(env, pi_solver, vi_solver, pi_path, vi_path, pi_time, vi_time)
    
    print("所有结果已保存到 results/ 目录")
    print("程序执行完成！")

def save_results(env, pi_solver, vi_solver, pi_path, vi_path, pi_time, vi_time):
    """保存结果到文件"""
    
    # 保存值函数
    with open("results/value_functions.txt", "w", encoding="utf-8") as f:
        f.write("值函数比较\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略迭代值函数:\n")
        for state in sorted(env.get_all_states()):
            f.write(f"状态 {state}: {pi_solver.V[state]:.6f}\n")
        
        f.write("\n值迭代值函数:\n")
        for state in sorted(env.get_all_states()):
            f.write(f"状态 {state}: {vi_solver.V[state]:.6f}\n")
        
        f.write("\n值函数差异:\n")
        for state in sorted(env.get_all_states()):
            if not env.is_terminal(state):
                diff = abs(pi_solver.V[state] - vi_solver.V[state])
                f.write(f"状态 {state}: {diff:.6f}\n")
    
    # 保存策略
    with open("results/policies.txt", "w", encoding="utf-8") as f:
        f.write("策略比较\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略迭代策略:\n")
        for state in sorted(env.get_all_states()):
            if not env.is_terminal(state):
                f.write(f"状态 {state}: {pi_solver.policy[state]}\n")
        
        f.write("\n值迭代策略:\n")
        for state in sorted(env.get_all_states()):
            if not env.is_terminal(state):
                f.write(f"状态 {state}: {vi_solver.policy[state]}\n")
    
    # 保存路径
    with open("results/paths.txt", "w", encoding="utf-8") as f:
        f.write("最优路径比较\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略迭代路径:\n")
        for i, state in enumerate(pi_path):
            f.write(f"步骤 {i}: {state}\n")
        
        f.write(f"\n路径长度: {len(pi_path)}\n")
        f.write(f"总奖励: {pi_solver.simulate_path()[1]:.4f}\n")
        
        f.write("\n值迭代路径:\n")
        for i, state in enumerate(vi_path):
            f.write(f"步骤 {i}: {state}\n")
        
        f.write(f"\n路径长度: {len(vi_path)}\n")
        f.write(f"总奖励: {vi_solver.simulate_path()[1]:.4f}\n")
    
    # 保存性能统计
    with open("results/performance.txt", "w", encoding="utf-8") as f:
        f.write("性能统计\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("策略迭代:\n")
        f.write(f"  计算时间: {pi_time:.4f}秒\n")
        f.write(f"  迭代次数: {len(pi_solver.value_history)}\n")
        f.write(f"  平均每轮用时: {pi_time/len(pi_solver.value_history):.4f}秒\n")
        
        f.write("\n值迭代:\n")
        f.write(f"  计算时间: {vi_time:.4f}秒\n")
        f.write(f"  迭代次数: {len(vi_solver.value_history)}\n")
        f.write(f"  平均每轮用时: {vi_time/len(vi_solver.value_history):.4f}秒\n")
        
        f.write(f"\n比较:\n")
        f.write(f"  时间比 (PI/VI): {pi_time/vi_time:.2f}\n")
        f.write(f"  迭代比 (PI/VI): {len(pi_solver.value_history)/len(vi_solver.value_history):.2f}\n")

if __name__ == "__main__":
    main()
