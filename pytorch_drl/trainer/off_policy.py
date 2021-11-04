#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from .base_trainer import BaseTrainer


class OffPolicyTrainer(BaseTrainer):

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

    def __call__(self):
        self.agent.train()
        for episode in range(self.num_episodes):
            # initialize the env and state
            state = self.env.reset()
            for step in itertools.count():
                # select an action
                action = self.agent.act(state)
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action)
                # buffer the experience
                self.agent.cache(state, action, reward, done, next_state)
                # learn from the experience
                self.agent.learn()
                # update the state
                state = next_state
                # check if end
                if done:
                    self.episode_duration.append(step + 1)
                    break
            if episode % self.agent.sync_step == 0:
                self.agent.sync_critic()
        self.env.close()
        return self.episode_duration
