#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch

from .base_env import BaseEnv

# nominal parameters from a maximum-likelihood fit
d_r = 0.0956
d_p = 0.0214
b_r = 0.0965
k_m = 0.0116

A_c = np.array([[-d_r, 0., 0.],
                [d_p + k_m, -d_p - k_m, 0.],
                [0., d_p, -d_p]])

B_c = np.array([[d_r, b_r],
                [0., 0.],
                [0., 0.]])

# sampling rate
T_s = 10

A = np.exp(A_c * T_s)

B = np.linalg.inv(A_c) @ (A - np.eye(A.shape[0])) @ B_c


class DiscreteCRN(BaseEnv):
    """
    s' = A @ s + B @ a
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self._trajectory = []
        self._steps_done = 0

    @property
    def state_dim(self):
        return A.shape[1]

    @property
    def discrete(self):
        return False

    @property
    def action_dim(self):
        # continuous action space
        return 1

    def reset(self):
        self._trajectory = []
        self._steps_done = 0
        observation = self.rng.uniform(0, 1, (self.state_dim, ))
        self._trajectory.append(observation)
        return observation

    def step(self, action):
        action = np.array([1., action])
        observation = A @ self._trajectory[self._steps_done] + B @ action
        reward = 0.
        done = False
        info = {}
        self._trajectory.append(observation)
        self._steps_done += 1
        return observation, reward, done, info

    def render(self):
        T = np.arange(0, T_s * (self._steps_done + 1), T_s)
        R, P, G = np.stack(self._trajectory, axis=1)
        plt.plot(T, R, 'o-', label='R')
        plt.plot(T, P, 'o-', label='P')
        plt.plot(T, G, 'o-', label='G')
        plt.xlabel('time [min]')
        plt.ylabel('intensity')
        plt.legend()
        plt.show()

    def close(self):
        self._trajectory = []
        self._steps_done = 0


class ContinuousCRN(BaseEnv):
    """
    ds / dt = A_c @ s + B_c @ a
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self._trajectory = []
        self._steps_done = 0

    @property
    def state_dim(self):
        return A_c.shape[1]

    @property
    def discrete(self):
        return False

    @property
    def action_dim(self):
        # continuous action space
        return 1

    def reset(self):
        self._trajectory = []
        self._steps_done = 0
        observation = self.rng.uniform(0, 1, (self.state_dim,))
        self._trajectory.append(observation)
        return observation

    @staticmethod
    def func(t, y, action):
        action = np.array([1., action])
        return A_c @ y + B_c @ action

    def step(self, action):
        delta = 0.2
        sol = solve_ivp(
            self.func,
            (0, T_s + delta),
            self._trajectory[self._steps_done],
            t_eval=np.arange(0, T_s + delta, delta),
            args=(action),
        )
        observation = sol.y[:, -1]
        reward = 0.
        done = False
        info = {}
        self._trajectory.append(observation)
        self._steps_done += 1
        return observation, reward, done, info

    def render(self):
        T = np.arange(0, T_s * (self._steps_done + 1), T_s)
        r, p, g = np.stack(self._trajectory, axis=1)
        plt.plot(T, R, 'o-', label='r')
        plt.plot(T, P, 'o-', label='p')
        plt.plot(T, G, 'o-', label='g')
        plt.xlabel('time [min]')
        plt.ylabel('intensity')
        plt.legend()
        plt.show()

    def close(self):
        self._trajectory = []
        self._steps_done = 0
