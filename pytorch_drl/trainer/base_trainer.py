#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class BaseTrainer(abc.ABC):

    def __init__(
        self,
        num_episodes,
        agent,
        env,
    ):
        self.num_episodes = num_episodes
        self.agent = agent
        self.env = env

    def __call__(self):
        raise NotImplementedError
