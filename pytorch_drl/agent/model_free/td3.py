#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .ddpg import DDPGAgent


class TD3Agent(DDPGAgent):
    """ Twin Delayed DDPG (TD3).

    "Addressing Function Approximation Error in Actor-Critic Methods" (2018). arxiv.org/abs/1802.09477
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
        policy_noise=0.2,
        burnin_size=32,
        learn_every=1,
        update_every=2,
        sync_every=2,
        sync_coefficient=0.005,
    ):
        # initialize critic Q(s, a), actor pi(s) and experience replay buffer R
        # initialize target critric Q' and target actor pi'
        # hyperparameters for `act`
        # hyperparameters for `learn`
        # step counter
        super().__init__(
            device,
            actor,
            critic,
            discount_factor,
            learning_rate,
            buffer_capacity,
            batch_size,
            exploration_noise,
            burnin_size,
            learn_every,
            sync_every,
            sync_coefficient,
        )
        self.policy_noise = policy_noise
        self.update_every = update_every

    def act(self, state):
        # explore, a = pi(s) + epsilon, epsilon ~ N(0, sigma)
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
        Q1, Q2 = self.critic.get_twin(state, action)
        with torch.no_grad():
            # compute a' = pi(s') + epsilon, epsilon ~ clip(N(0, sigma), -c, c)
            next_action = self.actor_target.act(next_state, self.policy_noise, noise_clipping=True, c=0.5)
            # compute Q(s', a') = min_{i = 1, 2} Q_{i}(s', a')
            next_Q = torch.min(*self.critic_target.get_twin(next_state, next_action))
            # compute expected Q(s, a) = r(s, a) + gamma * max_{a'} Q(s', a')
            Q_target = reward + self.gamma * next_Q
        # update critic by minimizing sum of loss of TD error
        critic_loss = self.critic_criterion(Q1, Q_target) + self.critic_criterion(Q2, Q_target)
        self._update_nn(self.critic_optim, critic_loss)
        # policy learning side of DDPG
        # update actor by maximizing E[Q_{1}(s, pi(s))]
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # in TD3, actor is updated less frequently than twin critics are, originally, `update_every` is equivalent to `sync_every`
        if self.curr_step % self.update_every == 0:
            self._update_nn(self.actor_optim, actor_loss)
        # sync weights of target networks every `sync_every` steps
        if self.curr_step % self.sync_every == 0:
            self._sync_weights(self.critic_target, self.critic, soft_updating=True, tau=self.tau)
            self._sync_weights(self.actor_target, self.actor, soft_updating=True, tau=self.tau)
        return {
            'loss/actor': actor_loss,
            'loss/critic': critic_loss,
            'Q': Q1.mean(),
        }
