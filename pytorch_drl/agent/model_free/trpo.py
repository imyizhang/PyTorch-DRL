#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch

from ..base_agent import BaseAgent


class TRPOAgent(BaseAgent):
    """Trust Region Policy Optimization (TRPO).

    "Trust Region Policy Optimization" (2015). arxiv.org/abs/1502.05477
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
        exploration_noise=0.1,
        burnin_size=32,
        learn_every=1,
        sync_every=1,
        sync_coefficient=0.005,
    ):
        # initialize critic Q(s, a), actor pi(s) and experience replay buffer R
        super().__init__(
            device,
            actor,
            critic,
            discount_factor,
            learning_rate,
            buffer_capacity,
            batch_size,
        )
        # initialize target critric Q' and target actor pi'
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        # hyperparameters for `act`
        self.exploration_noise = exploration_noise
        # hyperparameters for `learn`
        self.burnin_size = burnin_size
        self.learn_every = learn_every
        self.sync_every = sync_every
        self.tau = sync_coefficient
        # step counter
        self.curr_step = 0

    def train(self):
        self.actor.train()
        self.actor_target.eval()
        self.critic.train()
        self.critic_target.eval()

    def act(self, state):
        # explore
        action = self.actor.act(state, self.exploration_noise)
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
        # it's time to learn
        actor_loss, critic_loss = None, None
        # sample a random minibatch of transitions from experience replay buffer
        state, action, reward, done, next_state = self.recall()
        # Q learning side of DDPG
        # compute estimated Q(s, a)
        Q = self.critic(state, action)
        with torch.no_grad():
            # compute Q(s', a') = Q(s', pi(s'))
            next_Q = self.critic_target(next_state, self.actor_target(next_state))
            # compute expected Q(s, a) = r(s, a) + gamma * max_{a'} Q(s', a')
            Q_target = reward + self.gamma * next_Q
        # update critic by minimizing the loss of TD error
        critic_loss = self.critic_criterion(Q, Q_target)
        self._update_nn(self.critic_optim, critic_loss)
        # policy learning side of DDPG
        # update actor using the sampled policy gradient
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self._update_nn(self.actor_optim, actor_loss)
        # sync weights of target networks every `sync_every` steps
        if self.curr_step % self.sync_every == 0:
            self._sync_weights(self.critic_target, self.critic, soft_updating=True, tau=self.tau)
            self._sync_weights(self.actor_target, self.actor, soft_updating=True, tau=self.tau)
        return {
            'loss/actor': actor_loss,
            'loss/critic': critic_loss,
            'Q': Q.mean(),
        }
