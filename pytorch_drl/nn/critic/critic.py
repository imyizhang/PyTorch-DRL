#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..base_nn import BaseCritic
from ..approximator import MLPApproximator


class Critic(BaseCritic):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256, 256,),
        approximator_activation=torch.nn.Identity(),
        approximator=MLPApproximator,
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim + action_dim,
            1,
            approximator_dims,
            out_activation=approximator_activation,
        )

    def forward(self, state, action):
        return self.approximator(torch.cat((state, action), dim=-1))

    def configure_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()


class TwinCritic(BaseCritic):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        embedding_dims=(256,),
        approximator_dims=(256,),
        approximator_activation=torch.nn.Identity(),
        approximator=MLPApproximator,
    ):
        super().__init__()
        self.embedding = approximator(
            state_dim + action_dim,
            hidden_dim,
            embedding_dims,
            out_activation=torch.nn.ReLU(),
        )
        self.approximator1 = approximator(
            hidden_dim,
            1,
            approximator_dims,
            out_activation=approximator_activation,
        )
        self.approximator2 = approximator(
            hidden_dim,
            1,
            approximator_dims,
            out_activation=approximator_activation,
        )

    def forward(self, state, action):
        embedded = self.embedding(torch.cat((state, action), dim=-1))
        return self.approximator1(embedded)

    def get_twin(self, state, action):
        embedded = self.embedding(torch.cat((state, action), dim=-1))
        return self.approximator1(embedded), self.approximator2(embedded)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()
