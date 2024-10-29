import random
import gym
import numpy as np

from enviroment.domain.AP import AP
from enviroment.domain.sensing_device import SensingDevice
import config


class IIoTEnv(gym.Env):
    def __init__(self, config):
        super(IIoTEnv, self).__init__()
        self.config = config
        self.num_devices = self.config['sensing_device']['num_devices']
        self.num_aps = self.config['ap']['num_devices']
        self.aps = []
        self.devices = []
        for i in range(self.num_devices):
            device_type = random.choice(self.config['sensing_device']['device_types'])
            position = self.config['sensing_device']['position'][i]
            self.devices.append(SensingDevice(i, device_type, position))
        for i in range(self.num_aps):
            position = self.config['ap']['position'][i]
            self.aps.append(AP(i, position))

        self.action_space = self.config['CONFIG_ENVIRONMENT']['action_space']  # 0: local processing, 1: offload
        self.observation_space = self.config['CONFIG_ENVIRONMENT']['observation_space']

    def reset(self):
        self.ap.data_queue.clear()
        for device in self.devices:
            device.data_queue.clear()
            device.package_id = 0
        self.state = self._get_state()
        return self.state

    def step(self, action):
        # Convert action to binary representation for each device
        actions = bin(action)[2:].zfill(self.num_devices)
        actions = [int(a) for a in actions]

        for i, device in enumerate(self.devices):
            device.sense_data()  # Each device senses data at each step
            if actions[i] == 0:  # Local processing
                device.process_data()
            device.transmit_data(self.ap)

        self.ap.process_data()  # AP processes received data

        self.state = self._get_state()
        reward = self._calculate_reward()
        done = False  # Define your termination condition
        info = {}
        return self.state, reward, done, info

    def _get_state(self):
        device_queue_lengths = [len(device.data_queue) for device in self.devices]
        device_processing = [device.data_queue[0].process_required if device.data_queue else 0 for device in
                             self.devices]
        ap_queue_length = len(self.ap.data_queue)

        return np.array(device_queue_lengths + device_processing + [ap_queue_length], dtype=np.float32)

    def _calculate_reward(self):
        # Example reward function: minimize total queue lengths
        total_queue_length = sum(len(device.data_queue) for device in self.devices) + len(self.ap.data_queue)
        return -total_queue_length


if __name__ == "__main__":
    env = IIoTEnv(config)
    state = env.reset()

    for _ in range(10):  # Example interaction loop
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, State: {state}, Reward: {reward}, Next State: {next_state}")
        state = next_state
