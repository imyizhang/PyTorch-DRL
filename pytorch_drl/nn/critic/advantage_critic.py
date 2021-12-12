#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..base_nn import BaseCritic
from ..approximator import MLPApproximator


class AdvantageCritic(BaseCritic):

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
            1,
            approximator_dims,
            out_activation=approximator_activation,
        )

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()
