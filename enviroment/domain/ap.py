import collections

from config import config


class AP:
    def __init__(self, id, position, computing_capability):
        self.id = id
        self.data_queue = collections.deque()
        self.update_queue = collections.deque()
        self.position = position
        # self.distance_matrix = []
        self.computing_capability = computing_capability

    def receive_data(self, package):
        """
        接收来自SensingDevice的数据
        """
        self.data_queue.append(package)
        print(f"AP {self.id} received package {package.id}")

    def process_data(self):
        """
        数据处理方法：对队列中的data进行计算处理，使data_size减小
        """
        while self.data_queue:
            packet = self.data_queue.popleft()
            if packet.process_required:
                packet.timer += int((packet.data_size / self.computing_capability)/config['environment']['time_slot'])
                packet.data_size = int(packet.data_size * 0.5)
                print(f"AP {self.id} processed packet {packet.id}, new size: {packet.data_size}")
            self.update_queue.append(packet)
