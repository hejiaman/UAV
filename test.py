import torch

from agent.PPO_d import PolicyNet
from enviroment.uav_env import DroneSchedulingEnv

config = {
    'num_drones': 3,
    'user_positions': [[10,20], [30,40], [50,60], [70,80]],  # 用户坐标
    'coverage_radius': 50,
    'compute_power': 10,  # MIPS/step
    'bandwidth': 5,       # MB/step
    'transmit_power': 0.5,
    'max_queue_length': 5,
    'task_rate': 0.1,
    'max_steps': 100,
    'reward_weights': [0.4, 0.4, 0.2],  # 时延,能耗,负载
    'task_complete_bonus': 5,
    'compute_energy_coeff': 0.5,  # 必须添加
    'transmit_energy_coeff': 0.1  # 必须添加
}


def test_dynamic_dimensions():
    for drone_count in [2, 3, 4]:
        config.update({'num_drones': drone_count})
        env = DroneSchedulingEnv(config)
        state = env.reset()
        print(f"无人机数量={drone_count} | 状态维度={len(state)} | 示例值={state[:3]}...")

        # 验证网络兼容性
        dummy_input = torch.randn(1, len(state))
        model = PolicyNet(len(state), 128, env.action_space.n)
        try:
            output = model(dummy_input)
            print(f"网络测试通过，输出形状={output.shape}")
        except Exception as e:
            print(f"网络测试失败: {str(e)}")