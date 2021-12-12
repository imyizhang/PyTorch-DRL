#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
# on-policy
from .model_free.discrete_ppo import DiscretePPOAgent
