#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import collections


Experience = collections.namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'done', 'next_state')
)


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *transition):
        self.memory.append(Experience(*transition))

    def sample(self, batch_size):
        if batch_size > len(self):
            batch_size = len(self)
        transitions = random.sample(self.memory, batch_size)
        return Experience(*zip(*transitions))
