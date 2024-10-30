import collections
from enviroment.domain.packet import Packet

from config import config


class SensingDevice:
    def __init__(self, id, device_type, position, computing_capability, monitored_devices):
        self.id = id
        self.package_id = 0
        self.device_type = device_type
        self.data_queue = collections.deque()
        self.position = position
        # self.distance_matrix = []
        self.computing_capability = computing_capability
        self.monitored_devices = monitored_devices  # devices being monitored

    def sense_data(self):
        """
        数据感知方法：创建新的data并加入队列
        """
        # package create
        self.package_id += 1
        new_packet = Packet(self.package_id,
                             config['data']['data_process_mapping'][self.device_type],
                             self.device_type,
                             config['data']['data_size_mapping'][self.device_type],
                             self.id,
                             self.monitored_devices)
        # time cost
        new_packet.timer += config['sensing_task']['sensing_duration']

        self.data_queue.append(new_packet)
        print(f"Sensing device {self.id} sensed package {self.package_id}")

    def process_data(self):
        """
        数据处理方法：对队列中的data进行计算处理，使data_size减小
        """
        if self.data_queue:
            packet = self.data_queue[0]
            if packet.process_required:
                packet.timer += int((packet.data_size / self.computing_capability) / config['environment']['time_slot'])
                packet.data_size = int(packet.data_size * 0.5)
                print(f"Device {self.id} processed packet {packet.id}, new size: {packet.data_size}")

    def transmit_data(self, ap):
        """
        输出传送：从队列中移除data并移动到ap的数据队列中
        """
        if self.data_queue:
            transmission_rate = 0
            package = self.data_queue.popleft()
            package.timer += int((package.data_size / transmission_rate) / config['environment']['time_slot'])
            ap.receive_data(package)
            print(f"Device {self.id} transmitted package {package.id} to AP {ap.id}")
