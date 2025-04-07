# Barbara
# 开发时间：2025/4/3 11:21
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from agent.PPO_d import PPO
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

# 超参数
actor_lr = 3e-4
critic_lr = 1e-3
hidden_dim = 128
gamma = 0.99
lmbda = 0.95
epochs = 10
eps = 0.2
num_episodes = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 初始化环境和PPO
env = DroneSchedulingEnv(config)
initial_state = env.reset()

state_dim = len(initial_state)  # 动态获取实际维度

agent = PPO(
    state_dim=state_dim,
    hidden_dim=hidden_dim,
    action_dim=env.action_space.shape[0],
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    lmbda=lmbda,
    epochs=epochs,
    eps=eps,
    gamma=gamma,
    device=device
)


# 训练循环
return_list = []
for i in range(10):  # 10个训练阶段
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [], 'actions': [],
                'next_states': [], 'rewards': [], 'dones': []
            }

            state = env.reset()
            done = False

            while not done:
                state = env._get_state() if hasattr(env, '_get_state') else env.state
                action = agent.take_action(state, env)
                next_state, reward, done, _ = env.step(action)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            agent.update(transition_dict)

            if (i_episode + 1) % 5 == 0:
                pbar.set_postfix({
                    'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                    'return': f'{np.mean(return_list[-5:]):.1f}'
                })
            pbar.update(1)

# 结果可视化
plt.plot(return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on Drone Scheduling')
plt.show()