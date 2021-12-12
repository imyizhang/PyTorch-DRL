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

    def __init__(
        self,
        agent,
        update_coefficient=0,
        end_bs=32,
    ):
        super().__init__(
            agent,
            update_coefficient,
            end_bs,
        )
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
            pass

    def __call__(self):
        pass


class LinearBS(BSScheduler):

    def __call__(self):
        assert self.update_coefficient > 1
        self.curr_bs = self.end_er + (self.start_bs - self.end_er) * (1.0 - self.curr_step / self.update_coefficient)
        self.curr_er = min(self.end_er, int(self.curr_er))
