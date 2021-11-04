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
