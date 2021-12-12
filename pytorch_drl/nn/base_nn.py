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

    @abc.abstractmethod
    def act(self, state, action_noise):
        # return action
        raise NotImplementedError

    # return optimizer
    def configure_optimizer(self, lr):
        return None

    # return criterion
    def configure_criterion(self):
        return None


class BaseCritic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.approximator = approximator

    @abc.abstractmethod
    def forward(self, state, action):
        #return self.approximator(state)
        raise NotImplementedError

    # return optimizer
    def configure_optimizer(self, lr):
        return None

    # return criterion
    def configure_criterion(self):
        return None
