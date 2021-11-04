#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import torch

from .base_env import BaseEnv


class GymEnv(BaseEnv, gym.Wrapper):

    def __init__(self, env):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

    @property
    def state_space_shape(self):
        return self.env.observation_space.shape

    @property
    def discrete(self):
        # discrete action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return True
        # continuous action space
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return False
        else:
            raise RuntimeError

    @property
    def action_space_shape(self):
        # discrete action space
        if self.discrete:
            return self.env.action_space.n
        # continuous action space
        else:
            return self.env.action_space.shape

    def reset(self, dtype=torch.float32):
        observation = self.env.reset()
        state = torch.from_numpy(observation).to(dtype=dtype)
        return state

    def step(self, action, dtype=torch.float32):
        observation, reward, done, info = self.env.step(action)
        state = torch.from_numpy(observation).to(dtype=dtype)
        return state, reward, done, info

    def close(self):
        return self.env.close()
