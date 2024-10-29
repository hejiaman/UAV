import collections

from enviroment.domain.datapackage import DataPackage


class SensingDevice:
    def __init__(self, id, device_type, position):
        self.id = id
        self.device_type = device_type
        self.data_queue = collections.deque()

        self.position = position
        self.distance_matrix = []

    def sense_data(self):
        """
        数据感知方法：创建新的data并加入队列
        """
        new_data = DataPackage(data_id, process_required, self.data_type, data_size)
        self.data_queue.append(new_data)
        print(f"Device {self.id} sensed data {data_id}")

    def process_data(self):
        """
        数据处理方法：对队列中的data进行计算处理，使data_size减小
        """
        if self.data_queue:
            data = self.data_queue[0]
            if data.process_required:
                data.data_size = int(data.data_size * 0.5)
                print(f"Device {self.id} processed data {data.id}, new size: {data.data_size}")

    def transmit_data(self, ap):
        """
        输出传送：从队列中移除data并移动到ap的数据队列中
        """
        if self.data_queue:
            data = self.data_queue.popleft()
            ap.receive_data(data)
            print(f"Device {self.id} transmitted data {data.id} to AP {ap.id}")
