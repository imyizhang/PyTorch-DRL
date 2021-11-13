#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .crn import Wrapper, make


class CRNEnv(Wrapper):

    def __init__(
        self,
        device,
        env,
        dtype=None,
        **kwargs,
    ) -> None:
        self.device = device
        self.env = make(env, **kwargs) if isinstance(env, str) else env
        super().__init__(self.env)
        self.dtype = dtype

    @property
    def state(self):
        state = self.env.state
        state = torch.as_tensor(
            state,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.state_dim)
        return state

    def action_sample(self):
        action = self.env.action_sample()
        action = torch.as_tensor(
            action,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.action_dim)
        return action

    def reset(self):
        state = self.env.reset()
        state = torch.as_tensor(
            state,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.state_dim)
        return state

    def step(self, action):
        action = action.cpu().detach().item()
        state, reward, done, info = self.env.step(action)
        state = torch.as_tensor(
            state,
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
