#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .base_agent import BaseAgent
from pytorch_drl.net import DummyActor


class DummyAgent(BaseAgent):

    def __init__(
        self,
        device,
        actor,
        critic=None,
        discount_factor=None,
        buffer_capacity=None,
        batch_size=None,
        sync_step=None,
    ):
        super().__init__(
            device,
            actor,
            critic,
            discount_factor,
            buffer_capacity,
            batch_size,
            sync_step,
        )
        assert isinstance(self.actor, DummyActor)
        self.curr_step = 0

    def act(self):
        # explore
        action = self.actor.configure_sampler().to(self.device)
        # increment step
        self.curr_step += 1
        return action

    def train(self):
        return None

    def learn(self):
        loss = torch.zeros(1).to(self.device)
        return loss
