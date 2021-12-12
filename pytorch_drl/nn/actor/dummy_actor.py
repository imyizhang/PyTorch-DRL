#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ..base_nn import BaseActor


class DummyActor(BaseActor):

    def __init__(self):
        super().__init__()

    def forward(self, state):
        raise RuntimeError


class ConstantActor(DummyActor):

    def __init__(self, action_dim, value=None):
        super().__init__()
        # continuous action space
        self.action = torch.empty(size=(1, action_dim)).uniform_(0, 1) if value is None else torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device)


class RandomActor(DummyActor):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(self, state):
        # continuous action space
        action = torch.empty(size=(1, self.action_dim)).uniform_(0, 1)
        return action.to(device=state.device)


class DiscreteConstantActor(DummyActor):

    def __init__(self, action_dim, value=None):
        super().__init__()
        # discrete action space
        self.action = torch.randint(
            low=0,
            high=action_dim,
            size=(1, 1),
        ) if value is None else torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device)


class DiscreteRandomActor(DummyActor):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(
        self,
        state,
        p=None,
    ):
        # discrete action space
        if p is None:
            action = torch.randint(
                low=0,
                high=self.action_dim,
                size=(1, 1),
            )
        else:
            action = torch.multinomial(p.view(-1), 1).view(1, 1)
        return action.to(device=state.device)
