import collections
from enviroment.domain.package import Package

import config


class SensingDevice:
    def __init__(self, id, device_type, position):
        self.id = id
        self.package_id = 0
        self.device_type = device_type
        self.data_queue = collections.deque()

        self.position = position
        self.distance_matrix = []

    def sense_data(self):
        """
        数据感知方法：创建新的data并加入队列
        """
        new_data = Package(self.package_id,
                           config.CONFIG_DATA['data_process_mapping'][self.device_type],
                           self.device_type,
                           config.CONFIG_DATA['data_size_mapping'][self.device_type])
        self.package_id += 1
        self.data_queue.append(new_data)
        print(f"Device {self.id} sensed package {self.package_id}")

    def process_data(self):
        """
        数据处理方法：对队列中的data进行计算处理，使data_size减小
        """
        if self.data_queue:
            package = self.data_queue[0]
            if package.process_required:
                package.data_size = int(package.data_size * 0.5)
                print(f"Device {self.id} processed package {package.id}, new size: {package.data_size}")

    def transmit_data(self, ap):
        """
        输出传送：从队列中移除data并移动到ap的数据队列中
        """
        if self.data_queue:
            package = self.data_queue.popleft()
            ap.receive_data(package)
            print(f"Device {self.id} transmitted package {package.id} to AP {ap.id}")
