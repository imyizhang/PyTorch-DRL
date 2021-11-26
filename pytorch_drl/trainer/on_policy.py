#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from .base_trainer import BaseTrainer
from pytorch_drl.utils import EpisodeLogger


class OnPolicyTrainer(BaseTrainer):

    def __init__(
        self,
        env,
        agent,
        num_episodes=10,
    ):
        super().__init__(
            env,
            agent,
            num_episodes,
        )
        self.logger = EpisodeLogger()

    def __call__(self):
        # set the agent in training mode
        self.agent.train()
        for episode in range(self.num_episodes):
            # initialize the env and state
            state = self.env.reset()
            self.logger.reset(state)
            rewards = []
            for step in itertools.count():
                # select an action
                action = self.agent.act(state)
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action)
                # buffer the experience
                self.agent.cache(state, action, reward, done, next_state)
                # learn from the experience
                loss = self.agent.learn()
                # logging
                self.logger.step(self.env, state, action, reward, loss)
                print(episode, step, reward.item(), loss.item())
                # update the state
                state = next_state
                # check if end
                rewards.append(reward)
                if done or sum(rewards[-10:]) / len(rewards[-10:]) > 0.95:
                    break
            # logging
            self.logger.episode()
            if episode % self.agent.sync_step == 0:
                self.agent.sync_critic()
        self.env.close()
        return self.logger
