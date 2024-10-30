from config import config


class Packet:
    def __init__(self, id, process_required, data_type, data_size, data_sample_device, data_source):
        self.id = id
        self.process_required = process_required
        self.data_type = data_type
        self.data_size = data_size
        self.data_sample_device = data_sample_device  # sample data device: sensing device
        self.data_source = data_source  # generated data device: iiot device
        self.is_packet_updated = False
        self.timer = 0
