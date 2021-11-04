#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import torch

from pytorch_drl.utils import ReplayBuffer


class BaseAgent(abc.ABC):

    def __init__(
        self,
        device,
        actor,
        critic,
        discount_factor,
        buffer_capacity,
        batch_size,
        sync_step,
    ):
        self.device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optim = actor.configure_optimizer()
        self.actor_criterion = actor.configure_criterion()
        self.critic_optim = critic.configure_optimizer()
        self.critic_criterion = critic.configure_criterion()
        self.gamma = discount_factor
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.sync_step = sync_step

    @abc.abstractmethod
    def act(self):
        raise NotImplementedError

    def cache(self, state, action, reward, done, next_state):
        transition = (state, action, reward, done, next_state)
        self.buffer.push(*transition)

    def recall(self):
        batch = self.buffer.sample(self.batch_size)
        # state size -> (batch, state_dim)
        state = torch.cat(batch.state, dim=0)
        # action size -> (batch, action_dim)
        action = torch.cat(batch.action, dim=0)
        # reward size -> (batch, 1)
        reward = torch.cat(batch.reward, dim=0)
        # done size -> (batch, 1)
        done = torch.cat(batch.done, dim=0)
        # next_state size -> (batch, state_dim)
        next_state = torch.cat(batch.next_state, dim=0)
        return state, action, reward, done, next_state

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    def sync_critic(self):
        self.critic.load_state_dict(self.actor.state_dict())
