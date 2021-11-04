#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from collections import namedtuple, deque


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'done', 'next_state')
)


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return None
        transitions = random.sample(self.memory, batch_size)
        return Experience(*zip(*transitions))
