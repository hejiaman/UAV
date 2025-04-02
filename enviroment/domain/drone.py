# Barbara
# 开发时间：2025/4/1 21:38

import numpy as np

class Drone:
    def __init__(self, drone_id, position, config):
        """
        Args:
            drone_id: 无人机唯一标识
            position: [x, y] 坐标
            config: 全局配置字典
        """
        self.id = drone_id
        self.position = np.array(position)
        self.compute_power = config['compute_power']  # MIPS/step
        self.bandwidth = config['bandwidth']  # MB/step
        self.transmit_power = config['transmit_power']  # 传输功率系数
        self.max_queue_length = config['max_queue_length']
        self.task_queue = []
        self.current_task = None
        self.energy_consumed = 0  # 累计能耗

    def add_task(self, task):
        """添加任务到队列"""
        if len(self.task_queue) < self.max_queue_length:
            self.task_queue.append(task)
            return True
        return False

    def process_task(self):
        """处理当前任务"""
        if self.current_task is None and self.task_queue:
            self.current_task = self.task_queue.pop(0)

        if self.current_task:
            if self.current_task['status'] == 'transferring':
                # 完成传输（传输时延在转发时已计算）
                self.current_task['status'] = 'computing'

            elif self.current_task['status'] == 'computing':
                # 计算处理
                processed = min(self.compute_power, self.current_task['compute_req'])
                self.current_task['compute_req'] -= processed
                self.energy_consumed += processed * self.compute_energy

                if self.current_task['compute_req'] <= 0:
                    self.current_task['status'] = 'completed'
                    return True
        return False

    @property
    def compute_energy(self):
        """计算能耗系数（每MIPS能耗）"""
        return 0.5  # 可根据配置调整