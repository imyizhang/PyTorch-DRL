#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .crn import ContinuousTimeCRN


class Wrapper(ContinuousTimeCRN):

    def __init__(self, env):
        self.env = env

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def state(self):
        return self.env.state

    @property
    def discrete(self):
        return self.env.discrete

    @property
    def action_dim(self):
        return self.env.action_dim

    def action_sample(self):
        return self.env.action_sample()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def compute_reward(self, achieved_goal, desired_goal):
        return self.env.compute_reward(achieved_goal, desired_goal)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
