#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np


class Env(abc.ABC):

    def __init__(self):
        self._rng = np.random.RandomState(seed=None)

    def seed(self, seed: typing.Optional[int] = None):
        if seed is None:
            try:
                yield
            finally:
                pass
        else:
            state = self._rng.get_state()
            self._rng.seed(seed)
            try:
                yield
            finally:
                self._rng.set_state(state)

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: typing.Union[int, np.ndarray]):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self
