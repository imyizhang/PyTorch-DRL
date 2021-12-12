#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_agent import BaseAgent
from .dummy_agent import DummyAgent
# off-policy
from .model_free.dqn import DQNAgent
from .model_free.ddqn import DDQNAgent
from .model_free.prioritized_ddqn import PrioritizedDDQNAgent
from .model_free.dueling_ddqn import DuelingDDQNAgent
from .model_free.distributional_dqn import C51Agent
from .model_free.noisy_dqn import NoisyDQNAgent
from .model_free.a2c import A2CAgent
from .model_free.rainbow import RainbowAgent
from .model_free.clipped_ddqn import ClippedDDQNAgent
from .model_free.discrete_sac import DiscreteSACAgent
from .model_free.ddpg import DDPGAgent
from .model_free.sac import SACAgent
from .model_free.td3 import TD3Agent
# on-policy
from .model_free.discrete_ppo import DiscretePPOAgent
from .model_free.trpo import TRPOAgent
from .model_free.ppo import PPOAgent
# with discrete/continuous wrapper
from .discrete import *
from .continuous import *

__all__ = (
    'BaseAgent',
    'DummyAgent',
    'DQNAgent',
    'DDQNAgent',
    'PrioritizedDDQNAgent',
    'DuelingDDQNAgent',
    'C51Agent',
    'NoisyDQNAgent',
    'A2CAgent'
    'RainbowAgent',
    'ClippedDDQNAgent',
    'DiscreteSACActor',
    'DiscretePPOActor',
    'DDPG',
    'SACActor',
    'TD3Agent',
    'TRPOAgent',
    'PPOAgent',
)
