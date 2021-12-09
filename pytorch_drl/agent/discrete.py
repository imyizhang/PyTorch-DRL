#!/usr/bin/env python
# -*- coding: utf-8 -*-

# off-policy
from .model_free.dqn import DQNAgent
from .model_free.double_dqn import DoubleDQNAgent
from .model_free.discrete_sac import DiscreteSACAgent
# on-policy
from .model_free.discrete_ppo import DiscretePPOAgent
