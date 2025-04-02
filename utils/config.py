import yaml

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)

# 默认配置
DEFAULT_CONFIG = {
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
    'task_complete_bonus': 5
}