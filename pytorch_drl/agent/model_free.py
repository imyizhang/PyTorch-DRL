#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math

import torch

from .base_agent import BaseAgent


class DQNAgent(BaseAgent):

    def __init__(
        self,
        device,
        actor,
        critic,
        discount_factor=0.999,
        buffer_capacity=10000,
        batch_size=128,
        sync_step=10,
        exploration_rate=0.9,
        exploration_rate_min=0.05,
        exploration_rate_decay=200,
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
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
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
        # decrease exploration rate
        self._exploration_rate_scheduler()
        return action

    def _exploration_rate_scheduler(self):
        # exponential decay
        self.exploration_rate = self.exploration_rate_min + \
        (self.exploration_rate - self.exploration_rate_min) * \
        math.exp(-1. * self.curr_step / self.exploration_rate_decay)

    def train(self):
        self.actor.train()
        self.critic.eval()

    def learn(self):
        state, action, reward, done, next_state = self.recall()
        # compute Q(s, a)
        Q = self.actor(state).gather(dim=1, index=action)
        # compute V(s') := max_{a'} Q(s', a')
        with torch.no_grad():
            V = self.critic(next_state).max(dim=1, keepdim=True).values
        # compute expected Q(s, a) := r(s, a) + gamma * V(s')
        Q_expected = reward + self.gamma * V
        # optimize the actor
        self.actor_optim.zero_grad()
        loss = self.actor_criterion(Q, Q_expected)
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()
