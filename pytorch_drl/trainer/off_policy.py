#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_trainer import BaseTrainer


class OffPolicyTrainer(BaseTrainer):

    def __init__(
        self,
        num_episodes,
        agent,
        env,
    ):
        super().__init__(
            num_episodes,
            agent,
            env,
        )

    def __call__(self):
        for episode in range(self.num_episodes):
            # initialize the env and state
            state = self.env.reset()
            while True:
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
                    break
            if episode % self.agent.sync_step == 0:
                self.agent.sync_critic()
        self.env.close()
