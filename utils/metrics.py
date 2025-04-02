# Barbara
# 开发时间：2025/4/1 21:46
# utils/metrics.py
import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate_reward(env):
        """综合奖励计算"""
        # 1. 时延惩罚
        delay_penalty = MetricsCalculator.average_delay(env.completed_tasks)

        # 2. 能耗惩罚
        energy_penalty = sum(d.energy_consumed for d in env.drones) / 100  # 归一化

        # 3. 负载均衡惩罚
        load_penalty = MetricsCalculator.load_balance(env.drones)

        # 加权奖励（权重从配置读取）
        weights = env.config['reward_weights']
        reward = - (weights[0] * delay_penalty +
                    weights[1] * energy_penalty +
                    weights[2] * load_penalty)

        # 完成任务奖励
        reward += len(env.completed_tasks) * env.config['task_complete_bonus']
        return reward

    @staticmethod
    def average_delay(completed_tasks):
        """计算平均时延（包含传输时延）"""
        if not completed_tasks:
            return 0
        total_delay = sum(
            (t['complete_time'] - t['create_time'] + t.get('transfer_time', 0))
            for t in completed_tasks)
        return total_delay / len(completed_tasks)

    @staticmethod
    def load_balance(drones):
        """计算负载均衡指标（队列长度方差）"""
        queue_lengths = [len(d.task_queue) for d in drones]
        return np.var(queue_lengths)