#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from .base_trainer import BaseTrainer
from pytorch_drl.utils import EpisodeLogger


class OffPolicyTrainer(BaseTrainer):

    def __init__(
        self,
        env,
        agent,
        num_episodes=10,
        num_timesteps=None,
    ):
        super().__init__(
            env,
            agent,
            num_episodes,
            num_timesteps
        )
        self.logger = EpisodeLogger()

    def __call__(self, mode):
        # set the agent in training mode
        self.agent.train()
        for episode in range(self.num_episodes):
            # initialize the env and state
            state = self.env.reset()
            self.logger.reset(state)
            rewards = []
            timesteps = itertools.count() if self.num_timesteps is None else range(self.num_timesteps)
            for step in timesteps:
                # select an action
                action = self.agent.act(state)
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action, mode=mode)
                # buffer the experience
                self.agent.cache(state, action, reward, done, next_state)
                # learn from the experience
                loss = self.agent.learn()
                # logging
                self.logger.step(self.env, state, action, reward, info, loss)
                print(episode, step, reward.item(), loss.item())
                # update the state
                state = next_state
                # check if end
                rewards.append(reward)
                #if done or sum(rewards[-10:]) / len(rewards[-10:]) > 0.9:
                #    break
            # logging
            self.logger.episode()
            if episode % self.agent.sync_step == 0:
                self.agent.sync_critic()
        self.env.close()
        return self.logger
