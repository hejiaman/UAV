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
        self.compute_energy_coeff = config.get('compute_energy_coeff', 0.5)  # 计算能耗系数(J/MIPS)
        self.transmit_energy_coeff = config.get('transmit_energy_coeff', 0.1)  # 传输能耗系数(J/MB)
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
        """处理当前任务（仅计算能耗）"""
        if self.current_task is None and self.task_queue:
            self.current_task = self.task_queue.pop(0)

        if self.current_task:
            if self.current_task['status'] == 'transferring':
                # 完成传输（传输能耗已在转发时计算）
                self.current_task['status'] = 'computing'

            elif self.current_task['status'] == 'computing':
                # 计算处理能耗
                processed = min(self.compute_power, self.current_task['compute_req'])
                self.current_task['compute_req'] -= processed
                self.energy_consumed += processed * self.compute_energy_coeff

                if self.current_task['compute_req'] <= 0:
                    self.current_task['status'] = 'completed'
                    return True
        return False

    def forward_task(self, target_drone, task):
        """将当前任务转发给目标无人机（包含完整的时延和能耗计算）"""
        if not self.current_task:
            return False

        task = self.current_task
        distance = np.linalg.norm(self.position - target_drone.position)

        # 计算传输时延和能耗
        transfer_time = task['data_size'] / min(self.bandwidth, target_drone.bandwidth)
        transfer_energy = self.transmit_power * transfer_time

        # 更新任务状态
        task.update({
            'status': 'transferring',
            'transfer_from': self.id,
            'transfer_to': target_drone.id,
            'transfer_time': transfer_time,
            'transfer_energy': transfer_energy
        })

        # 转移任务并记录能耗
        self.current_task = None
        target_drone.task_queue.append(task)
        self.energy_consumed += transfer_energy
        return True