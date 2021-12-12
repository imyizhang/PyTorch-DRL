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
        self.capacity = int(capacity)
        self.memory = collections.deque([], maxlen=int(capacity))

    def __len__(self):
        return len(self.memory)

    def push(self, *transition):
        self.memory.append(Experience(*transition))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Experience(*zip(*transitions))
