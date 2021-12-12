#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_nn import BaseActor, BaseCritic
from .actor.dummy_actor import DummyActor, ConstantActor, RandomActor, DiscreteConstantActor, DiscreteRandomActor
from .actor.actor import Actor
from .actor.sac_actor import SACActor, DiscreteSACActor
from .actor.ppo_actor import PPOActor, DiscretePPOActor
from .critic.q_critic import QCritic, TwinQCritic
from .critic.critic import Critic, TwinCritic
from .critic.advantage_critic import AdvantageCritic
# with discrete/continuous wrapper
from .discrete import *
from .continuous import *

__all__ = (
    'BaseActor',
    'BaseCritic',
    'DummyActor',
    'DiscreteConstantActor',
    'DiscreteRandomActor',
    'DiscreteSACActor',
    'DiscretePPOActor',
    'QCritic',
    'TwinQCritic',
    'ConstantActor',
    'RandomActor',
    'Actor',
    'SACActor',
    'PPOActor',
    'Critic',
    'TwinCritic',
)
