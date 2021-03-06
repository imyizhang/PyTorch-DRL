#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random

import torch

from ..base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q Network (DQN).

    "Human-Level Control Through Deep Reinforcement Learning" (2015). nature.com/articles/nature14236
    """

    def __init__(
        self,
        device,
        actor,
        critic,
        discount_factor=0.99,
        learning_rate=1e-3,
        buffer_capacity=10000,
        batch_size=32,
        exploration_rate=0.1,
        burnin_size=32,
        learn_every=1,
        sync_every=4,
    ):
        # initialize critic Q(s, a) and experience replay buffer R
        super().__init__(
            device,
            actor,
            critic,
            discount_factor,
            learning_rate,
            buffer_capacity,
            batch_size,
        )
        # initialize target critric Q'
        self.critic_target = copy.deepcopy(self.critic)
        # hyperparameters for `act`
        self.eps_threshold = exploration_rate
        # hyperparameters for `learn`
        self.burnin_size = burnin_size
        self.learn_every = learn_every
        self.sync_every = sync_every
        # step counter
        self.curr_step = 0

    def train(self):
        self.critic.train()
        self.critic_target.eval()

    def act(self, state):
        # explore, epsilon-greedy policy
        eps = random.random()
        if eps < self.eps_threshold:
            action = self.actor.act(state)
        # exploit, a = pi^{*}(s) = argmax_{a} Q^{*}(s, a)
        else:
            with torch.no_grad():
                action = self.critic(state).max(dim=1, keepdim=True).indices
        # step
        self.curr_step += 1
        return action

    def learn(self):
        # at least `burnin_size` transitions buffered before learning
        if self.curr_step < self.burnin_size:
            return None
        # learn every `learn_every` steps
        if self.curr_step % self.learn_every != 0:
            return None
        # sync weights of target networks every `sync_every` steps
        if self.curr_step % self.sync_every == 0:
            self._sync_weights(self.critic_target, self.critic)
        # it's time to learn
        actor_loss, critic_loss = None, None
        # sample a random minibatch of transitions from experience replay buffer
        state, action, reward, done, next_state = self.recall()
        # Q learning
        # compute estimated Q^{*}(s, a)
        Q = self.critic(state).gather(dim=1, index=action)
        with torch.no_grad():
            # compute Q^{*}(s', a')
            next_Q = self.critic_target(next_state).max(dim=1, keepdim=True).values
            # compute expected Q^{*}(s, a) = r(s, a) + gamma * max_{a'} Q^{*}(s', a')
            Q_target = reward + self.gamma * next_Q
        # update critic by minimizing the loss of TD error
        critic_loss = self.critic_criterion(Q, Q_target)
        self._update_nn(self.critic_optim, critic_loss)
        return {
            'loss/actor': actor_loss,
            'loss/critic': critic_loss,
            'Q': Q.mean(),
        }
