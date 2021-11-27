#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(
        self,
        device,
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
        self.curr_step = 0

    def act(self, state):
        action = self.actor.configure_sampler().to(self.device)
        # increment step
        self.curr_step += 1
        return action

    def train(self):
        # to be filled

    def learn(self):
        # to be filled
