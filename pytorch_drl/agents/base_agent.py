#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class BaseAgent(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def take_action(self):
        raise NotImplementedError

    @abc.abstractmethod
    def explore_env(self):
        raise NotImplementedError
