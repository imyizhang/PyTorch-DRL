#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .base_net import BaseActor, BaseCritic


class QNet(torch.nn.Module):

    def __init__(self, approximator):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features, outputs)
        )

    def forward(self, state):
        return self.approximator(state)


class QNetActor(BaseActor):

    def __init__(self):
        super().__init__()
        self.approximator = QNet()

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self):
        return torch.optim.RMSprop(self.parameters())


class QNetCritic(BaseCritic):

    def __init__(self):
        super().__init__()
        self.approximator = QNet()

    def forward(self, state):
        return self.approximator(state)

    def configure_optimizer(self):
        return None
