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
        learning_rate,
        buffer_capacity,
        batch_size,
    ):
        self.device = device
        self.actor = actor.to(device) if actor is not None else None
        self.critic = critic.to(device) if critic is not None else None
        self.gamma = discount_factor if discount_factor is not None else None
        self.actor_optim = actor.configure_optimizer(learning_rate) if (actor is not None) and (learning_rate is not None) else None
        self.actor_criterion = actor.configure_criterion() if actor is not None else None
        self.critic_optim = critic.configure_optimizer(learning_rate) if (actor is not None) and (learning_rate is not None) else None
        self.critic_criterion = critic.configure_criterion() if critic is not None else None
        self.buffer = ReplayBuffer(buffer_capacity) if buffer_capacity is not None else None
        self.batch_size = batch_size if batch_size is not None else None

    def cache(self, state, action, reward, done, next_state):
        transition = (state, action, reward, done, next_state)
        self.buffer.push(*transition)

    def recall(self):
        batch = self.buffer.sample(self.batch_size)
        # state -> (batch, state_dim)
        state = torch.cat(batch.state, dim=0)
        # action -> (batch, action_dim)
        action = torch.cat(batch.action, dim=0)
        # reward -> (batch, 1)
        reward = torch.cat(batch.reward, dim=0)
        # done -> (batch, 1)
        done = torch.cat(batch.done, dim=0)
        # next_state -> (batch, state_dim)
        next_state = torch.cat(batch.next_state, dim=0)
        return state, action, reward, done, next_state

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    @staticmethod
    def _update_nn(
        optimizer,
        loss,
        gradient_clipping=False,
        nn=None,
        c=1.0,
    ):
        optimizer.zero_grad()
        loss.backward()
        if gradient_clipping:
            for param in nn.parameters():
                param.grad.data.clamp_(min=-c, max=c)
        optimizer.step()

    @staticmethod
    def _sync_weights(
        nn_target,
        nn,
        soft_updating=False,
        tau=1e-3,
    ):
        # weights of target networks are updated by having them slowly track the learned networks with tau << 1
        if soft_updating:
            for param_target, param in zip(nn_target.parameters(), nn.parameters()):
                param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)
        # updated by directly copying the weights
        else:
            nn_target.load_state_dict(nn.state_dict())
