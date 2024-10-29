import collections


class AP:
    def __init__(self, id, position):
        self.id = id
        self.data_queue = collections.deque()

        self.position = position
        self.distance_matrix = []

    def receive_data(self, data):
        """
        接收来自SensingDevice的数据
        """
        self.data_queue.append(data)
        print(f"AP {self.id} received data {data.id}")

    def process_data(self):
        """
        数据处理方法：对队列中的data进行计算处理，使data_size减小
        """
        if self.data_queue:
            data = self.data_queue.popleft()
            if data.process_required:
                data.data_size = int(data.data_size * 0.5)
                print(f"AP {self.id} processed data {data.id}, new size: {data.data_size}")