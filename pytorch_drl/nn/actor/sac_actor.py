#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..base_nn import BaseActor
from ..approximator import MLPApproximator


class SACActor(BaseActor):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256,),
        approximator_activation=torch.nn.Sigmoid(),
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

    def act(
        self,
        state,
        action_noise,
        noise_clipping=False,
        c=0.5,
    ):
        with torch.no_grad():
            # action, continuous action space
            action = self(state)
            # noise
            noise = torch.randn_like(action) * action_noise
            # noise clipping
            if noise_clipping:
                noise = torch.clamp(noise, min=-c, max=c)
            # add noise
            action += noise
            # action clipping
            action = torch.clamp(action, min=0.0, max=1.0)
            return action

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class DiscreteSACActor(BaseActor):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256,),
        approximator_activation=torch.nn.Sigmoid(),
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

    def act(
        self,
        state,
        action_noise,
        noise_clipping=False,
        c=0.5,
    ):
        with torch.no_grad():
            # action, continuous action space
            action = self(state)
            # noise
            noise = torch.randn_like(action) * action_noise
            # noise clipping
            if noise_clipping:
                noise = torch.clamp(noise, min=-c, max=c)
            # add noise
            action += noise
            # action clipping
            action = torch.clamp(action, min=0.0, max=1.0)
            return action

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
