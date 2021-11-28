#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random
import math

import torch

from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):

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
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
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
        Q = self.critic(state, action)
        with torch.no_grad():
            next_Q = self.critic_target(next_state, self.actor_target(next_state))
        Q_expected = reward + mask * next_Q
        self.critic_optim.zero_grad()
        loss = self.critic_criterion(Q, Q_expected)
        loss.backward()
        self.critic_optim.step()
        for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            param_target.data.copy_(param_target.data * tau + tar.data * (1.0 - tau))
        return loss
