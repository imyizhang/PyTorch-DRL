#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import contextlib
import typing

import numpy as np


@contextlib.contextmanager
def temp_seed(
    rng: np.random.RandomState,
    seed: typing.Optional[int] = None,
):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class Env(abc.ABC):

    def __init__(self):
        self._rng = np.random.RandomState(seed=None)
        self._seed = None

    def seed(self, seed: typing.Optional[int] = None):
        self._seed = seed

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: typing.Union[int, np.ndarray]):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, mode: str = 'human'):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self
