import gym
from gym import spaces

import messenger.envs.config as config

class MessengerEnv(gym.Env):
    '''
    Base Messenger class that defines the action and observation spaces.
    '''

    def __init__(self):
        super().__init__()
        # up, down, left, right, stay
        self.action_space = spaces.Discrete(len(config.ACTIONS))

        # observations, not including the text manual
        self.observation_space = spaces.Dict({
            "entities": spaces.Box(
                low=0,
                high=14,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 3)
            ),
            "avatar": spaces.Box(
                low=15,
                high=16,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 1)
            )
        })

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError