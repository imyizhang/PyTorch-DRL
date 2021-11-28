#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .crn import Wrapper, make


class CRNEnv(Wrapper):

    def __init__(
        self,
        env,
        device,
        dtype=torch.float32,
        **kwargs,
    ):
        self.env = make(env, **kwargs) if isinstance(env, str) else env
        super().__init__(self.env)
        self.device = device
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

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def discrete(self):
        return self.env.discrete

    @property
    def action_dim(self):
        return self.env.action_dim

    def action_sample(self):
        action = self.env.action_sample()
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
        state = self.env.reset()
        state = torch.as_tensor(
            state,
            dtype=self.dtype,
            device=self.device
        ).view(1, self.state_dim)
        return state

    def step(self, action, mode):
        # discrete action space
        if self.discrete:
            action = action.cpu().detach().item()
        # continuous action space
        else:
            action = action.view(-1).cpu().detach().numpy()
        state, reward, done, info = self.env.step(action, mode)
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
