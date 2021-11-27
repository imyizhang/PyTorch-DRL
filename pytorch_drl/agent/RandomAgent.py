#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(
        self,
        exploration_rate=0.9,
    ):
        self.exploration_rate = exploration_rate
        self.curr_step = 0

    def act(self, state):
        # explore
        if random.random() < self.exploration_rate:
            action = self.actor.configure_sampler().to(self.device)
        # exploit
        else:
            with torch.no_grad():
                action = self.actor(state).max(dim=1, keepdim=True).indices
        # increment step
        self.curr_step += 1
        return action

    def train(self):
        # to be filled

    def learn(self):
        # to be filled
