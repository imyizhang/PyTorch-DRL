#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random
import math

import torch

from ..base_agent import BaseAgent


class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient (DDPG).

    "Continuous Control with Deep Reinforcement Learning" (2015). arxiv.org/pdf/1509.02971.pdf
    """

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
        self.critic.train()

    def learn(self):
        state, action, reward, done, next_state = self.recall()
        # Q Learning side of DDPG
        Q = self.critic(state, action)
        with torch.no_grad():
            next_Q = self.critic_target(next_state, self.actor_target(next_state))
        Q_expected = reward + mask * next_Q
        # update critic
        critic_loss = self.critic_criterion(Q, Q_expected)
        self._update_network(self.critic_optimizer, critic_loss)
        # update target critic
        self._update_target_network(self.critic_target, self.critic, tau)
        # policy learning side of DDPG
        self.critic(state, self.actor(state)).mean()
        # update actor
        self._update_network(self.actor_optimizer, actor_loss)
        # update target actor
        self._update_target_network(self.actor_target, self.actor, tau)
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
        }

    @staticmethod
    def _update_network(optim, obj):
        optim.zero_grad()
        obj.backward
        optim.step()

    @staticmethod
    def _update_target_network(net_target, net, tau):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)
