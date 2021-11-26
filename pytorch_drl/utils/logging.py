#!/usr/bin/env python
# -*- coding: utf-8 -*-


class EpisodeLogger:

    def __init__(self):

        self.episode_trajectory = []
        self.episode_actions = []
        self.episode_duration = []
        self.episode_reward = []
        self.episode_loss = []
        self._init()

    @property
    def trajectories(self):
        return self.episode_trajectory

    @property
    def actions(self):
        return self.episode_actions

    @property
    def durations(self):
        return self.episode_duration

    @property
    def rewards(self):
        return self.episode_reward

    @property
    def losses(self):
        return self.episode_loss

    def _init(self):
        self._trajectory = []
        self._actions_taken = []
        self._steps_done = 0
        self._reward_aggregator = 0.0
        self._loss_aggregator = 0.0

    def reset(self, state):
        _state = state.view(-1).cpu().detach().numpy()
        self._trajectory.append(_state)

    def step(self, env, state, action, reward, loss):
        # torch.tensor -> numpy.ndarray
        _state = state.view(-1).cpu().detach().numpy()
        if env.discrete:
            _action = action.cpu().detach().item()
            _action = (_action + 1) / env.action_dim
        else:
            _action = action.view(-1).cpu().detach().numpy()
        _reward = reward.cpu().detach().item()
        _loss =  loss.cpu().detach().item()
        # step
        self._trajectory.append(_state)
        self._actions_taken.append(_action)
        self._steps_done += 1
        self._reward_aggregator += _reward
        self._loss_aggregator += _loss

    def episode(self):
        self.episode_trajectory.append(self._trajectory)
        self.episode_actions.append(self._actions_taken)
        self.episode_duration.append(self._steps_done)
        self.episode_reward.append(self._reward_aggregator / self._steps_done)
        self.episode_loss.append(self._loss_aggregator / self._steps_done)
        self._init()
