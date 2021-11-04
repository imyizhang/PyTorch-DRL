#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

from .utils import ReplayBuffer


class BaseAgent(abc.ABC):

    def __init__(
        self,
        device,
        buffer_capacity,
        batch_size,
        sync_step,
        discount_factor,
        actor,
        critic,
    ):
        self.device = device
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.sync_step = sync_step
        self.gamma = discount_factor
        self.actor = actor.to(device)
        self.actor.train()
        self.critic = critic.to(device)
        self.critic.eval()
        self.actor_optim = actor.configure_optimizer()
        self.actor_criterion = actor.configure_criterion()
        self.critic_optim = critic.configure_optimizer()
        self.critic_criterion = critic.configure_criterion()

    @abc.abstractmethod
    def act(self):
        raise NotImplementedError

    def cache(self, *args):
        self.buffer.push(*args)

    def recall(self):
        experience = self.buffer.sample(self.batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    def sync_critic(self):
        self.critic.load_state_dict(self.actor.state_dict())
