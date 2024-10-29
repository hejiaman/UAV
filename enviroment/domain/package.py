class Package:
    def __init__(self, id, process_required, data_type, data_size):
        self.id = id
        self.process_required = process_required
        self.data_type = data_type
        self.data_size = data_size
        self.timer = 0
