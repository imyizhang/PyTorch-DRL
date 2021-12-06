#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from .base_trainer import BaseTrainer
from pytorch_drl.utils import EpisodeLogger


class DummyTrainer(BaseTrainer):

    def __init__(
        self,
        env,
        agent,
        num_episodes=1,
        num_timesteps=None,
    ):
        super().__init__(
            env,
            agent,
            num_episodes,
            num_timesteps
        )
        self.logger = EpisodeLogger()

    def __call__(self, reward_func):
        # set the agent in training mode
        self.agent.train()
        for episode in range(self.num_episodes):
            # initialize the env and state
            state = self.env.reset()
            self.logger.reset(state)
            timesteps = itertools.count() if self.num_timesteps is None else range(self.num_timesteps)
            for step in timesteps:
                # select an action
                action = self.agent.act()
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action, reward_func=reward_func)
                # buffer the experience
                #self.agent.cache(state, action, reward, done, next_state)
                # learn from the experience
                loss = self.agent.learn()
                # step logging
                self.logger.step(self.env, state, action, reward, info, loss)
                print(episode, step, reward.item(), loss.item())
                # update the state
                state = next_state
                # check if end
                #if done
                #    break
            # episode logging
            self.logger.episode()
        self.env.close()
        return self.logger
