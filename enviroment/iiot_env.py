import random
import gym
import numpy as np

from domain.ap import AP
from domain.sensing_device import SensingDevice
from domain.device import Device
from config import config


class IIoTEnv(gym.Env):
    """
        环境类
        ！！！ 感知 + 计算 + 传输 time_slot < time_unit，即当前感知任务完成后存在空档 ！！！
        aoi需计算空档时间
    """

    def __init__(self, config):
        super(IIoTEnv, self).__init__()
        self.state = None
        self.timer = 0

        self.config = config
        self.num_sensing_devices = self.config['sensing_device']['num_devices']
        self.num_aps = self.config['ap']['num_devices']
        self.num_devices = self.config['devices']['num_devices']

        self.aoi = np.zeros(self.num_devices)

        # Initialize AP and sensing devices
        self.ap = None
        self.sensing_devices = []
        self.devices = []

        # Init devices
        for i in range(self.num_devices):
            self.devices.append(Device(i))

        # Init sensing devices
        for i in range(self.num_sensing_devices):
            device_type = random.choice(self.config['sensing_device']['device_types'])
            position = self.config['sensing_device']['positions'][i]
            computing_capability = self.config['sensing_device']['computing_capability']
            monitored_devices = random.sample(self.devices, 3)
            self.sensing_devices.append(
                SensingDevice(i, device_type, position, computing_capability, monitored_devices))

        # Init ap
        self.ap = AP(0, self.config['ap']['positions'], self.config['ap']['computing_capability'])

        # Define action and observation spaces based on configuration
        self.action_space = gym.spaces.MultiDiscrete([2, self.num_aps])  # 0: local processing, 1: offload to an AP
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(2 * self.num_sensing_devices + self.num_aps,), dtype=np.float32
        )

    def reset(self):
        self.state = None
        # Reset the environment and all device/AP states
        for device in self.sensing_devices:
            device.data_queue.clear()
            device.package_id = 0
            device.device_type = random.choice(self.config['sensing_device']['device_types'])

        self.ap.data_queue.clear()

        self.state = self._get_state()
        return self.state

    def step(self, actions):
        # sensing device
        for i, device in enumerate(self.sensing_devices):
            device.sense_data()

            # LP or Op
            if actions[i][0] == 0:  # Local processing
                device.process_data()
            else:  # Offload to an AP
                ap_id = actions[i][1]
                if 0 <= ap_id < self.num_aps:
                    device.transmit_data(ap_id)

        self.ap.process_data()

        self.timer += self.config['environment']['time_unit']
        self.aoi += self.timer

        # aoi update
        aoi_reduction = np.full(self.num_devices, np.iinfo(np.int32).max)
        for packet in self.ap.update_queue:
            for device in packet.data_source:
                if packet.timer < aoi_reduction[device.id]:
                    aoi_reduction[device.id] = packet.timer
        self.ap.update_queue.clear()
        self.aoi = aoi_reduction

        self.state = self._get_state()
        reward = self._calculate_reward()
        done = False
        info = {}

        return self.state, reward, done, info

    def _get_state(self):
        # Construct the state vector based on the queues and processing requirements
        device_queue_lengths = [len(device.data_queue) for device in self.sensing_devices]
        device_processing_required = [
            device.data_queue[0].process_required if device.data_queue else 0 for device in self.sensing_devices
        ]
        ap_queue_lengths = [len(ap.data_queue) for ap in self.aps]

        return np.array(device_queue_lengths + device_processing_required + ap_queue_lengths, dtype=np.float32)

    def _calculate_reward(self):
        # Reward function that minimizes the total queue lengths
        total_queue_length = sum(len(device.data_queue) for device in self.sensing_devices) + sum(
            len(ap.data_queue) for ap in self.aps)
        return -total_queue_length  # Reward is negative of the total queue length to encourage minimizing queues


if __name__ == "__main__":
    env = IIoTEnv(config)
    state = env.reset()

    for _ in range(10):  # Example interaction loop
        actions = [[random.choice([0, 1]), random.randint(0, env.num_aps - 1)] for _ in range(env.num_sensing_devices)]
        next_state, reward, done, info = env.step(actions)

        print(f"Actions: {actions}, State: {state}, Reward: {reward}, Next State: {next_state}")
        state = next_state
