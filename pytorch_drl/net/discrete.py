#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .base_net import BaseActor, BaseCritic


class MLP(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class QNetActor(BaseActor):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=MLP,
        approximator_dim=256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.approximator = approximator(state_dim, action_dim, approximator_dim)

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self):
        return torch.optim.RMSprop(self.parameters())

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()

    def configure_sampler(self):
        # discrete action space
        return torch.randint(
            low=0,
            high=self.action_dim,
            size=(1, 1),
        )


class QNetCritic(BaseCritic):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=MLP,
        approximator_dim=256,
    ):
        super().__init__()
        self.approximator = approximator(state_dim, action_dim, approximator_dim)

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self):
        return None

    def configure_criterion(self):
        return None
