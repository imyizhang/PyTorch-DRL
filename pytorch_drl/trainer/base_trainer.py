#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class BaseTrainer(abc.ABC):

    def __init__(
        self,
        env,
        agent,
        num_episodes,
        num_timesteps,
    ):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps

    def __call__(self):
        raise NotImplementedError
