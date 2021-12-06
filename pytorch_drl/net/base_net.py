#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import torch


class BaseActor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.approximator = approximator

    @abc.abstractmethod
    def forward(self, state):
        #return self.approximator(state)
        raise NotImplementedError

    # return optimizer
    @abc.abstractmethod
    def configure_optimizer(self):
        raise NotImplementedError

    # return criterion
    @abc.abstractmethod
    def configure_criterion(self):
        raise NotImplementedError


class DummyActor(BaseActor):

    def __init__(self):
        super().__init__()

    def forward(self, state):
        raise RuntimeError

    def configure_optimizer(self):
        return None

    def configure_criterion(self):
        return None

    # return action sampler
    @abc.abstractmethod
    def configure_sampler(self):
        raise NotImplementedError


class BaseCritic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.approximator = approximator

    @abc.abstractmethod
    def forward(self, state):
        #return self.approximator(state)
        raise NotImplementedError

    # return optimizer
    @abc.abstractmethod
    def configure_optimizer(self):
        raise NotImplementedError

    # return criterion
    @abc.abstractmethod
    def configure_criterion(self):
        raise NotImplementedError
