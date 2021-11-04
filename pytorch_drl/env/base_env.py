#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class BaseEnv(abc.ABC):

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
