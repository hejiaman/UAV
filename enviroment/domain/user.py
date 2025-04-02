# Barbara
# 开发时间：2025/4/1 22:29
import numpy as np

class User:
    def __init__(self, user_id, position, task_rate=0.1):
        """
        Args:
            user_id: 用户唯一标识
            position: [x, y] 坐标
            task_rate: 每个时间步生成任务的概率
        """
        self.id = user_id
        self.position = np.array(position)
        self.task_rate = task_rate
        self.assigned_drone = None  # 被分配的无人机ID

    def generate_task(self, current_step):
        """以概率task_rate生成新任务"""
        if np.random.rand() < self.task_rate:
            return {
                'user_id': self.id,
                'position': self.position.copy(),
                'data_size': np.random.uniform(5, 20),  # MB
                'compute_req': np.random.uniform(10, 50),  # MIPS
                'deadline': current_step + np.random.randint(5, 15),
                'create_time': current_step,
                'status': 'pending',  # pending -> transferring -> computing -> completed
                'transfer_from': None,
                'transfer_to': None,
                'transfer_time': 0
            }
        return None