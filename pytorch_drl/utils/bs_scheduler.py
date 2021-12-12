#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import math


class BSScheduler(abc.ABC):

    def __init__(
        self,
        agent,
        update_coefficient=0,
        end_bs=32,
    ):
        if not hasattr(agent, 'batch_size'):
            raise RuntimeError
        self.agent = agent
        self.update_coefficient = update_coefficient
        self.start_bs = getattr(agent, 'batch_size')
        self.end_bs = end_bs
        self.curr_bs = self.start_bs
        # intrinsic step counter
        self.curr_step = 0

    def __call__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def reset(self):
        self.curr_step = 0

    def step(self):
        # step
        self.curr_step += 1
        # update current batch_size, self.curr_bs, with intrinsic step
        self()
        # update self.agent.batch_size
        setattr(self.agent, 'batch_size', self.curr_bs)


class ConstantBS(BSScheduler):

    def __call__(self):
        # batch
        if self.start_bs == -1:
            if not hasattr(self.agent, 'buffer'):
                raise RuntimeError
            self.curr_bs = len(getattr(self.agent, 'buffer'))
        # stochastic
        elif self.start_bs == 1:
            pass
        # minibatch
        else:
            if not hasattr(self.agent, 'buffer'):
                raise RuntimeError
            self.curr_bs = len(getattr(self.agent, 'buffer'))
            self.curr_bs = min(self.start_bs, self.curr_bs)


class LinearBS(BSScheduler):

    def __call__(self):
        assert self.update_coefficient > 1
        self.curr_bs = self.end_bs + (self.start_bs - self.end_bs) * (1.0 - self.curr_step / self.update_coefficient)
        self.curr_bs = min(self.end_bs, int(self.curr_bs))
