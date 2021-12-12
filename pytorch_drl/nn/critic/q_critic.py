#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..base_nn import BaseCritic
from ..approximator import MLPApproximator


class QCritic(BaseCritic):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256,),
        approximator_activation=torch.nn.Identity(),
        approximator=MLPApproximator,
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_dims,
            out_activation=approximator_activation,
        )

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
        #return torch.optim.RMSprop(self.parameters())

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()


class TwinQCritic(BaseCritic):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256,),
        approximator_activation=torch.nn.Identity(),
        approximator=MLPApproximator,
        param_sharing=False,
        hidden_dim=256,
        embedding_dims=(256,),
    ):
        super().__init__()
        self.param_sharing = param_sharing
        if param_sharing:
            self.embedding = approximator(
                state_dim,
                hidden_dim,
                embedding_dims,
                out_activation=torch.nn.ReLU(),
            )
            self.approximator1 = approximator(
                hidden_dim,
                action_dim,
                approximator_dims,
                out_activation=approximator_activation,
            )
            self.approximator2 = approximator(
                hidden_dim,
                action_dim,
                approximator_dims,
                out_activation=approximator_activation,
            )
        else:
            self.approximator1 = approximator(
                state_dim,
                action_dim,
                approximator_dims,
                out_activation=approximator_activation,
            )
            self.approximator2 = approximator(
                state_dim,
                action_dim,
                approximator_dims,
                out_activation=approximator_activation,
            )

    def forward(self, state):
        embedded = self.embedding(state) if self.param_sharing else state
        return self.approximator1(embedded)

    def get_twin(self, state):
        embedded = self.embedding(state) if self.param_sharing else state
        return self.approximator1(embedded), self.approximator2(embedded)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()
