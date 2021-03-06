#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class EpisodeLogger:

    def __init__(self):

        self.episode_trajectory = []
        self.episode_observation = []
        self.episode_action = []
        self.episode_duration = []
        self.episode_reward = []
        self.episode_state_in_tolerance = []
        self.episode_loss = []
        self._init()

    @property
    def trajectories(self):
        return self.episode_trajectory

    @property
    def observations(self):
        return self.episode_observation

    @property
    def actions(self):
        return self.episode_action

    @property
    def rewards(self):
        return self.episode_reward

    @property
    def durations(self):
        return self.episode_duration

    @property
    def tolerance(self):
        return self.episode_state_in_tolerance

    @property
    def losses(self):
        return self.episode_loss

    def _init(self):
        self._trajectory = []
        self._observations = []
        self._actions = []
        self._rewards = []
        self._steps_done = 0
        self._state_in_tolerance_aggregator = 0
        self._loss_aggregator = 0.0

    def reset(self):
        # state
        state = np.ones((3,))  # R = P = G = 1
        self._trajectory.append(state)
        # observation
        observation = state[[2]]  # G = 1
        self._observations.append(observation)

    def step(self, env, action, state, reward, info, losses):
        # torch.tensor -> numpy.ndarray
        if env.discrete:
            _action = action.cpu().detach().item()
            _action = (_action + 1) / env.action_dim
        else:
            _action = action.view(-1).cpu().detach().numpy()
        #_state = state.view(-1).cpu().detach().numpy()
        _reward = reward.cpu().detach().item()
        _loss = losses['loss/critic'].cpu().detach().item() if losses is not None else 0
        # step
        self._trajectory.append(info['state'])
        self._observations.append(info['observation'])
        self._actions.append(_action)
        self._rewards.append(_reward)
        self._steps_done += 1
        self._state_in_tolerance_aggregator += info['in_tolerance']
        self._loss_aggregator += _loss

    def episode(self):
        self.episode_trajectory.append(self._trajectory)
        self.episode_observation.append(self._observations)
        self.episode_action.append(self._actions)
        self.episode_reward.append(self._rewards)
        self.episode_duration.append(self._steps_done)
        self.episode_state_in_tolerance.append(self._state_in_tolerance_aggregator / self._steps_done * 100)
        self.episode_loss.append(self._loss_aggregator / self._steps_done)
        self._init()

    def plot(self):
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(6, 5),
            gridspec_kw={'height_ratios': [1, 1]}
        )
        #fig.tight_layout()
        axs[0].plot([sum(r) for r in self.rewards], marker='.', color='tab:orange')
        axs[0].set_ylabel('reward')
        axs[0].grid(True)
        axs[1].plot(self.tolerance, marker='.', color='tab:red')
        axs[1].set_ylim([0, 100])
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('% states in tolerance zone')
        axs[1].grid(True)
        plt.show()

    def plot_loss(self):
        plt.plot(self.losses, marker='.', color='tab:blue')
        plt.xlabel('Episode')
        plt.ylabel('loss')
        plt.grid(True)
        plt.show()
