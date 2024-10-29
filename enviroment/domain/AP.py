import collections


class AP:
    def __init__(self, id, position):
        self.id = id
        self.data_queue = collections.deque()

        self.position = position
        self.distance_matrix = []

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
        if self.data_queue:
            package = self.data_queue.popleft()
            if package.process_required:
                package.data_size = int(package.data_size * 0.5)
                print(f"AP {self.id} processed package {package.id}, new size: {package.data_size}")