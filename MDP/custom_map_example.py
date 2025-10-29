"""
地图自定义示例
演示如何使用二维数组自定义迷宫地图
"""
import numpy as np
from maze_env import MazeEnvironment
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from visualizer import Visualizer

def create_custom_maps():
    """创建几个自定义地图示例"""
    
    # 示例1：简单直线路径
    simple_map = np.array([
        [2, 0, 0, 0, 0, 0, 0, 0],  # 起始点在(0,0)
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 3]   # 目标点在(7,7)
    ])
    
    # 示例2：复杂迷宫
    complex_map = np.array([
        [2, 0, 1, 0, 0, 0, 0, 0],  # 起始点在(0,0)，墙壁在(0,2)
        [0, 1, 1, 1, 0, 1, 0, 0],  # 多个墙壁
        [0, 0, 0, 1, 0, 1, 0, 0],  # 部分墙壁
        [1, 1, 0, 1, 0, 0, 0, 0],  # 左侧墙壁
        [0, 0, 0, 0, 0, 0, 1, 0],  # 右侧墙壁
        [0, 1, 1, 0, 1, 1, 1, 0],  # 中间墙壁
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 3]   # 目标点在(7,7)
    ])
    
    # 示例3：螺旋迷宫
    spiral_map = np.array([
        [2, 0, 0, 0, 0, 0, 0, 0],  # 起始点在(0,0)
        [0, 1, 1, 1, 1, 1, 1, 0],  # 上方墙壁
        [0, 0, 0, 0, 0, 0, 1, 0],  # 右侧墙壁
        [0, 1, 1, 1, 1, 0, 1, 0],  # 内部墙壁
        [0, 1, 0, 0, 0, 0, 1, 0],  # 内部墙壁
        [0, 1, 0, 1, 1, 1, 1, 0],  # 内部墙壁
        [0, 0, 0, 0, 0, 0, 0, 0],  # 全路径
        [0, 0, 0, 0, 0, 0, 0, 3]   # 目标点在(7,7)
    ])
    
    return {
        "简单地图": simple_map,
        "复杂迷宫": complex_map,
        "螺旋迷宫": spiral_map
    }

def test_custom_map(map_name, custom_map):
    """测试自定义地图"""
    print(f"\n{'='*50}")
    print(f"测试地图: {map_name}")
    print(f"{'='*50}")
    
    # 创建环境
    env = MazeEnvironment(custom_map)
    visualizer = Visualizer(env)
    
    # 显示地图信息
    env.print_map()
    
    # 可视化地图
    fig, ax = env.visualize_maze(f"{map_name} - 迷宫环境")
    
    # 运行策略迭代
    print(f"\n运行策略迭代算法...")
    pi_solver = PolicyIteration(env, gamma=0.9, theta=1e-6)
    pi_V, pi_policy = pi_solver.solve()
    
    # 运行值迭代
    print(f"运行值迭代算法...")
    vi_solver = ValueIteration(env, gamma=0.9, theta=1e-6)
    vi_V, vi_policy = vi_solver.solve()
    
    # 模拟路径
    pi_path, pi_reward = pi_solver.simulate_path()
    vi_path, vi_reward = vi_solver.simulate_path()
    
    print(f"\n结果比较:")
    print(f"策略迭代 - 路径长度: {len(pi_path)}, 总奖励: {pi_reward:.2f}")
    print(f"值迭代   - 路径长度: {len(vi_path)}, 总奖励: {vi_reward:.2f}")
    
    # 可视化结果
    visualizer.plot_value_function(pi_V, f"{map_name} - 策略迭代值函数")
    visualizer.plot_policy(pi_policy, f"{map_name} - 策略迭代策略")
    visualizer.plot_path(pi_path, f"{map_name} - 策略迭代路径")
    
    return env, pi_solver, vi_solver

def main():
    """主函数"""
    print("地图自定义示例")
    print("="*60)
    print("地图格式说明:")
    print("0 = 路径")
    print("1 = 墙壁")
    print("2 = 起始点")
    print("3 = 目标点")
    print("="*60)
    
    # 获取自定义地图
    custom_maps = create_custom_maps()
    
    # 测试每个地图
    results = {}
    for map_name, custom_map in custom_maps.items():
        try:
            env, pi_solver, vi_solver = test_custom_map(map_name, custom_map)
            results[map_name] = {
                'env': env,
                'pi_solver': pi_solver,
                'vi_solver': vi_solver
            }
        except Exception as e:
            print(f"测试 {map_name} 时出错: {e}")
    
    # 显示如何创建自己的地图
    print(f"\n{'='*60}")
    print("如何创建自己的地图:")
    print(f"{'='*60}")
    print("""
# 示例：创建一个8x8的地图
my_map = np.array([
    [2, 0, 0, 0, 0, 0, 0, 0],  # 第0行：起始点在(0,0)
    [0, 1, 0, 0, 0, 0, 0, 0],  # 第1行：墙壁在(1,1)
    [0, 0, 0, 0, 0, 0, 0, 0],  # 第2行：全路径
    [0, 0, 0, 0, 0, 0, 0, 0],  # 第3行：全路径
    [0, 0, 0, 0, 0, 0, 0, 0],  # 第4行：全路径
    [0, 0, 0, 0, 0, 0, 0, 0],  # 第5行：全路径
    [0, 0, 0, 0, 0, 0, 0, 0],  # 第6行：全路径
    [0, 0, 0, 0, 0, 0, 0, 3]   # 第7行：目标点在(7,7)
])

# 使用自定义地图创建环境
env = MazeEnvironment(my_map)

# 或者修改现有环境的地图
env.set_custom_map(my_map)
    """)
    
    return results

if __name__ == "__main__":
    results = main()
