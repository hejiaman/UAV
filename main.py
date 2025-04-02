from enviroment.drone_env import DroneSchedulingEnv
from utils.config import load_config, DEFAULT_CONFIG
from stable_baselines3 import PPO
import os


def train():
    # 加载配置（可替换为从文件读取）
    config = DEFAULT_CONFIG.copy()
    config.update({
        'user_positions': [[i * 10, i * 20] for i in range(10)],  # 10个用户
        'num_drones': 4
    })

    # 创建环境
    env = DroneSchedulingEnv(config)

    # 训练PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/drone_scheduling/",
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64
    )
    model.learn(total_timesteps=100000)

    # 保存模型
    os.makedirs("./models", exist_ok=True)
    model.save("./models/drone_scheduler_ppo")
    print("Training completed and model saved")


if __name__ == "__main__":
    train()