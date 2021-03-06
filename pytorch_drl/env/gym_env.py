#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import torch


class GymEnv(gym.Wrapper):

    def __init__(
        self,
        env,
        device,
        dtype=torch.float32,
    ):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)
        self.device = device
        self.dtype = dtype

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

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
    def action_dim(self):
        # discrete action space
        if self.discrete:
            return self.env.action_space.n
        # continuous action space
        else:
            return self.env.action_space.shape[0]

    def action_sample(self):
        action = self.env.action_space.sample()
        # discrete action space
        if self.discrete:
            action = torch.as_tensor(
                action,
                dtype=self.dtype,
                device=self.device
            ).view(1, 1)
        # continuous action space
        else:
            action = torch.as_tensor(
                action,
                dtype=self.dtype,
                device=self.device
            ).view(1, self.action_dim)
        return action

    def reset(self):
        observation = self.env.reset()
        state = torch.as_tensor(
            observation,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.state_dim)
        return state

    def step(self, action):
        # discrete action space
        if self.discrete:
            action = action.cpu().detach().item()
            action = int(action)
        # continuous action space
        else:
            action = action.view(-1).cpu().detach().numpy()
        observation, reward, done, info = self.env.step(action)
        state = torch.as_tensor(
            observation,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.state_dim)
        reward = torch.as_tensor(
            reward,
            dtype=self.dtype,
            device=self.device
        ).view(1, 1)
        done = torch.as_tensor(
            done,
            dtype=torch.bool,
            device=self.device
        ).view(1, 1)
        return state, reward, done, info
