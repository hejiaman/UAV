import gym
from gym import spaces
import numpy as np


class IIoTEnv(gym.Env):
    def __init__(self, config):
        super(IIoTEnv, self).__init__()
        self.action_space = spaces.Discrete(config['action_space'])
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(config['observation_space'],), dtype=np.float32
        )
        # Initialize other environment components

    def reset(self):
        # Reset state of the environment
        return self.state

    def step(self, action):
        # Apply action, calculate reward, and update state
        return next_state, reward, done, {}
