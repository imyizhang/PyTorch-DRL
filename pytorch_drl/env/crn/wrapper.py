#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing
import numpy as np
from .env import Env

class Wrapper(Env):

    def __init__(self, env: Env):
        self.env = env

    def seed(self, seed: typing.Optional[int] = None):
        return self.env.seed(seed=seed)

    def reset(self):
        return self.env.reset()

    def step(self, action: typing.Union[int, np.ndarray], **kwargs):
        return self.env.step(action, **kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped
