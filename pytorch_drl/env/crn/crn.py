#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from .env import Env
from .ref_trajectory import ConstantRefTrajectory


def make(cls: str, **kwargs):
    if cls == 'CRN':
        return ContinuousTimeDiscreteActionCRN(**kwargs)
    elif cls == 'CRNContinuous':
        return ContinuousTimeContinuousActionCRN(**kwargs)
    elif cls == 'StochasticCRN':
        return StochasticContinuousTimeDiscreteActionCRN(**kwargs)
    else:
        raise RuntimeError


# refer to https://www.nature.com/articles/ncomms12546.pdf
# continuous-time fold-change model:
#     ds / dt = A_c @ s + B_c @ a
# with nominal parameters from a maximum-likelihood fit
# d_r = 0.0956
# d_p = 0.0214
# k_m = 0.0116
# b_r = 0.0965
#
# A_c = np.array([[-d_r, 0.0, 0.0],
#                 [d_p + k_m, -d_p - k_m, 0.0],
#                 [0.0, d_p, -d_p]])
#
# B_c = np.array([[d_r, b_r],
#                 [0.0, 0.0],
#                 [0.0, 0.0]])
#
# C = np.array([[0.0, 0.0, 1.0]])

# refer to https://static-content.springer.com/esm/art%3A10.1038%2Fncomms12546/MediaObjects/41467_2016_BFncomms12546_MOESM1324_ESM.pdf
# equivalent discrete-time fold-change model:
#     s' = A @ s + B @ a
# with experimental observation sampling rate
# T_s = 10
#
# A = np.exp(A_c * T_s)
#
# B = np.linalg.inv(A_c) @ (A - np.eye(A.shape[0])) @ B_c
#
# C = np.array([[0.0, 0.0, 1.0]])

d_r = 0.0956
d_p = 0.0214
k_m = 0.0116
b_r = 0.0965


class ContinuousTimeDiscreteActionCRN(Env):
    """
    ds / dt = A_c @ s + B_c @ a
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
        system_noise: float = 1e-3,
        theta: np.ndarray = np.array([d_r, d_p, k_m, b_r]),
        observation_mode: str = 'partially_observed',
    ) -> None:
        super().__init__()
        # reference trajectory generator
        self.ref_trajectory = ref_trajectory
        # sampling rate
        self._T_s = sampling_rate
        # observation noise
        self._observation_noise = observation_noise
        # action noise
        self._action_noise = action_noise
        # system noise
        self._system_noise = system_noise
        # parameters for continuous-time fold-change model
        self._theta = theta
        self._d_r, self._d_p, self._k_m, self._b_r = self._theta
        self._A_c = np.array([[-self._d_r, 0.0, 0.0],
                              [self._d_p + self._k_m, -self._d_p - self._k_m, 0.0],
                              [0.0, self._d_p, -self._d_p]])
        self._B_c = np.array([[self._d_r, self._b_r],
                              [0.0, 0.0],
                              [0.0, 0.0]])
        # observation mode, either noise corrupted G (and t) or perfect R, P, G (and t)
        # would be observed by an agent
        self._observation_mode = observation_mode
        # initialize
        self._init()

    def _init(self):
        # actions input
        self._actions = []
        # noise corrupted actions taken
        self._actions_taken = []
        # perfect R, P, G (and t) states
        self._trajectory = []
        # noise corrupted G (and t)
        self._observations = []
        # reward measuring the distance between perfect G and reference trajectory
        self._rewards = []
        # steps done
        self._steps_done = 0

    @property
    def state(self) -> typing.Optional[typing.List[np.ndarray]]:
        if self._trajectory and self._observations:
            # noise corrupted G (and t) observed
            if self._observation_mode == 'partially_observed':
                return self._observations[self._steps_done]
            # perfect R, P, G (and t) observed
            return self._trajectory[self._steps_done]
        return None

    @property
    def state_dim(self) -> int:
        # noise corrupted G (and t) observed
        if self._observation_mode == 'partially_observed':
            return 1  # G
        # perfect R, P, G (and t) observed
        return 3  # R, P, G

    @property
    def discrete(self) -> bool:
        return True

    @property
    def action_dim(self) -> int:
        # discrete action space
        return 20

    def action_sample(self) -> int:
        # discrete action space
        return self._rng.randint(0, self.action_dim)

    def reset(self) -> np.ndarray:
        self._init()
        # state
        state = np.ones((3,))  # R = P = G = 1
        self._trajectory.append(state)
        # observation
        observation = state[[2]]  # G = 1
        self._observations.append(observation)
        # noise corrupted G (and t) observed
        if self._observation_mode == 'partially_observed':
            return observation
        # perfect R, P, G (and t) observed
        return state

    def _observe(self, state):
        # G
        observation = state[[2]]
        # noise corrupted G
        observation += self._rng.normal(0.0, self._observation_noise)
        # clip G to [0, inf)
        observation = np.clip(observation, 0.0, np.inf)
        return observation

    def _func(self, t: float, y: np.ndarray, action: float) -> np.ndarray:
        a = np.array([1.0, action])
        return self._A_c @ y + self._B_c @ a

    def step(
        self,
        action: typing.Union[float, np.ndarray],
        reward_func: str,
    ):
        if self.state is None:
            raise RuntimeError
        # action
        if self.discrete:
            action = (action + 1) / self.action_dim  # float
        else:
            action = action[0]  # float
        self._actions.append(action)
        # action taken
        action += self._rng.normal(0.0, self._action_noise)
        action = np.clip(action, 0.0, 1.0)
        self._actions_taken.append(action)
        # state
        delta = 0.1  # system dynamics simulation sampling rate
        sol = solve_ivp(
            self._func,
            (0, self._T_s + delta),
            self._trajectory[self._steps_done],
            t_eval=np.arange(0, self._T_s + delta, delta),
            args=(action,),
        )
        state = sol.y[:, -1]
        state += self._rng.normal(0.0, self._system_noise)
        state = np.clip(state, 0.0, np.inf)
        self._trajectory.append(state)
        # observation
        observation = self._observe(state)
        self._observations.append(observation)
        # reward
        T = np.array([self._steps_done * self._T_s])
        reference = self.ref_trajectory(T)[0]
        reward = self._compute_reward(state[2], reference[0], reward_func)
        self._rewards.append(reward)
        # done
        done = False
        # info
        info = {'tolerance': self._compute_reward(state[2], reference[0], 'tolerance')}
        # noise corrupted G (and t) observed
        if self._observation_mode == 'partially_observed':
            info['state'] = state
            #info['observation'] = observation
        # perfect R, P, G (and t) observed
        else:
            #info['state'] = state
            info['observation'] = observation
        # step
        self._steps_done += 1
        # noise corrupted G (and t) observed
        if self._observation_mode == 'partially_observed':
            return observation, reward, done, info
        # perfect R, P, G (and t) observed
        return state, reward, done, info

    def _compute_reward(
        self,
        achieved_goal: float,
        desired_goal: float,
        func: str
    ) -> typing.Union[float, int]:
        tolerance = self.ref_trajectory.tolerance
        abs_diff = abs(desired_goal - achieved_goal)
        alpha = 0.5
        if func == 'negative_square':
            reward = -abs_diff ** 2
        elif func == 'negative_abs':
            reward = -abs_diff
        elif func == 'negative_logabs':
            reward = -np.log(abs_diff)
        elif func == 'negative_expabs':
            reward = -np.exp(abs_diff)
        elif func == 'inverse_abs':
            reward = 1. / abs_diff
        elif func == 'percentage':
            reward = 1. - (abs_diff / desired_goal) ** alpha
        elif func == 'tolerance':
            reward = 1 if (abs_diff / desired_goal < tolerance) else 0
        elif func == 'percentage_tolerance':
            reward = (1. - (abs_diff / desired_goal) ** alpha) if (abs_diff / desired_goal < tolerance) else 0.
        else:
            raise RuntimeError
        return reward

    def render(
        self,
        render_mode: str = 'human',
        actions: typing.Optional[typing.List] = None,
        trajectory: typing.Optional[typing.List] = None,
        observations: typing.Optional[typing.List] = None,
        rewards: typing.Optional[typing.List] = None,
        steps_done: typing.Optional[int] = None
    ) -> None:
        replay = not ((not actions) and (not trajectory) and (not observations) and (not rewards) or (not steps_done))
        if (self.state is None) and (not replay):
            raise RuntimeError
        # for replay
        # actions input
        _actions = actions if replay else self._actions
        # noise corrupted actions taken, unknown for replay
        _actions_taken = None if replay else self._actions_taken
        # perfect R, P, G states
        _trajectory = trajectory if replay else self._trajectory
        # noise corrupted G observation
        _observations = observations if replay else self._observations
        # reward measuring the distance between perfect G and reference trajectory
        _rewards = rewards if replay else self._rewards
        # steps done
        _steps_done = steps_done if replay else self._steps_done
        # data
        # simulation sampling rate
        delta = 0.1
        # reference trajectory and tolerance margin
        t = np.arange(0, self._T_s * _steps_done + delta, delta)
        ref_trajectory, tolerance_margin = self.ref_trajectory(t)
        # sfGFP
        T = np.arange(0, self._T_s * _steps_done + self._T_s, self._T_s)
        R, P, G = np.stack(_trajectory, axis=1)
        # fluorescent sfGFP observed
        G_observed = np.concatenate(_observations, axis=0)
        # intensity
        t_u = np.concatenate([
            np.arange(self._T_s * i, self._T_s * (i + 1) + 1) for i in range(_steps_done)
        ])
        u = np.array(_actions).repeat(self._T_s + 1) * 100
        u_applied = np.array(_actions_taken).repeat(self._T_s + 1) * 100 if _actions_taken is not None else None
        # reward
        reward = np.array(_rewards)
        # plot
        if render_mode == 'human':
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # subplot fluorescent sfGFP
            axs[0].plot(t, ref_trajectory, '--', color='grey')
            axs[0].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0].plot(T, G, 'o-', label='G', color='green')
            axs[0].plot(T, G_observed, 'o--', label='G observed', color='green', alpha=0.5)
            axs[0].set_ylabel('fluorescent sfGFP (1/min)')
            axs[0].legend(framealpha=0.2)
            # subplot intensity
            axs[1].plot(t_u, u, '-', label='u')
            if u_applied is not None:
                axs[1].plot(t_u, u_applied, '--', label='u applied', alpha=0.5)
            axs[1].set_xlabel('Time (min)')
            axs[1].set_ylabel('intensity (%)')
            axs[1].legend(framealpha=0.2)
            plt.show()
        else:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # subplot sfGFP
            axs[0, 0].plot(t, ref_trajectory, '--', color='grey')
            axs[0, 0].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0, 0].plot(T, R, 'o-', label='R', color='red')
            axs[0, 0].plot(T, P, 'o-', label='P', color='blue')
            axs[0, 0].plot(T, G, 'o-', label='G', color='green')
            #axs[0, 0].plot(T, G_observed, 'o--', label='G observed', color='green', alpha=0.5)
            axs[0, 0].set_ylabel('sfGFP (1/min)')
            axs[0, 0].legend(framealpha=0.2)
            # subplot intensity
            axs[1, 0].plot(t_u, u, '-', label='u')
            if u_applied is not None:
                axs[1, 0].plot(t_u, u_applied, '--', label='u applied', alpha=0.5)
            axs[1, 0].set_xlabel('Time (min)')
            axs[1, 0].set_ylabel('intensity (%)')
            axs[1, 0].legend(framealpha=0.2)
            # subplot fluorescent sfGFP
            axs[0, 1].plot(t, ref_trajectory, '--', color='grey')
            axs[0, 1].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0, 1].plot(T, G, 'o-', label='G', color='green')
            axs[0, 1].plot(T, G_observed, 'o--', label='G observed', color='green', alpha=0.5)
            axs[0, 1].set_ylabel('fluorescent sfGFP (1/min)')
            axs[0, 1].legend(framealpha=0.2)
            # subplot reward
            axs[1, 1].plot(T[1:], reward)
            axs[1, 1].set_xlabel('Time (min)')
            axs[1, 1].set_ylabel('reward')
            plt.show()

    def close(self) -> None:
        self._init()


class ContinuousTimeContinuousActionCRN(ContinuousTimeDiscreteActionCRN):
    """
    s' = A @ s + B @ a
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
        system_noise: float = 1e-3,
        theta: np.ndarray = np.array([d_r, d_p, k_m, b_r]),
        observation_mode: str = 'partially_observed',
    ) -> None:
        super().__init__(
            ref_trajectory,
            sampling_rate,
            observation_noise,
            action_noise,
            system_noise,
            theta,
            observation_mode,
    )

    @property
    def discrete(self) -> bool:
        return False

    @property
    def action_dim(self) -> int:
        # continuous action space
        return 1

    def action_sample(self) -> np.ndarray:
        # continuous action space
        return self._rng.uniform(0, 1, (self.action_dim,))


class StochasticContinuousTimeDiscreteActionCRN(ContinuousTimeDiscreteActionCRN):
    """
    S(t) = S(0) + Y1 + Y2 + Y3
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
        system_noise: float = 1e-3,
        theta: np.ndarray = np.array([d_r, d_p, k_m, b_r]),
        observation_mode: str = 'partially_observed',
    ) -> None:
        super().__init__(
            ref_trajectory,
            sampling_rate,
            observation_noise,
            action_noise,
            system_noise,
            theta,
            observation_mode,
    )

    def step(
        self,
        action: typing.Union[float, np.ndarray],
        reward_func: str,
    ):
        if self.state is None:
            raise RuntimeError
        # action
        if self.discrete:
            action = (action + 1) / self.action_dim  # float
        else:
            action = action[0]  # float
        self._actions.append(action)
        # action taken
        action += self._rng.normal(0.0, action_noise)
        action = np.clip(action, 0.0, 1.0)
        self._actions_taken.append(action)
        # state
        ####
        # A maximum number of steps to run before breaking.
        maxi = self._T_s*10**5

        d_r = 0.0956 ## these 4 model parameters may be entered as parameters
        d_p = 0.0214
        k_m = 0.0116
        b_r = 0.0965

        # Define all rate parameters in the model.
        # Each column is a reaction vector: \zeta_k = y_k' - y_k.
        # Total number of reaction channels in the model.
        k = np.array([b_r*action, b_p, d_r, d_p, d_p, k_m, k_m])
        zeta = np.array([[ 0,  0,  0,  0,  0,  0,  0],
                         [+1,  0, -1,  0,  0,  0,  0],
                         [ 0, +1,  0, -1,  0, -1, +1],
                         [ 0,  0,  0,  0, -1, +1, -1]])
        R = zeta.shape[1]

        # initialize: x1 free gene, x2 mRNA, x3 immature proteins, x4 mature proteins.
        # A vector holding the jump times of the trajectory.
        # A matrix whose columns gives the state of the model at the jump times.
        # Initialize the state.
        # a vector holding the intensities of the reaction channels.
        # vectors for the integrated intensity functions and the next jump times of the unit Poisson processes.
        # vector holding actual times before next jumps in each reaction channel.
        initial = np.array([random.randint(1,3), random.randint(1, 20), random.randint(0,10), random.randint(0,10)])
        T = np.zeros(maxi)
        sol = np.zeros((zeta.shape[0],maxi))
        sol[:,0] = initial
        lamb = np.zeros(R)
        T_k = np.zeros(R)
        P_k = np.zeros(R)
        t = np.zeros(R)

        # uniform random variables in order to find the first jump time of each unit Poisson process.
        # set first jump time of each unit Poisson process.
        r_k = self._rng.rand(R)
        P_k = np.log(1. / r_k)

        for i in range(0, maxi):

            # Set values for the intensity functions.
            lamb[0] = k[0]*sol[0,i]
            lamb[1] = k[1]*sol[1,i]
            lamb[2] = k[2]*sol[1,i]
            lamb[3] = k[3]*sol[2,i]
            lamb[4] = k[4]*sol[3,i]
            lamb[5] = k[5]*sol[2,i]
            lamb[6] = k[6]*sol[3,i]

            # Find the amount of time required for each reaction channel to fire
            # (under the assumption no other channel fires first)
            # Find the index of the where the minimum is achieved.
            for c in range(0,R):
                if lamb[c] != 0:
                    t[c] = (Pk[c] - Tk[c])/lamb[c]
                else:
                    t[c] = np.inf

            loc = 0
            for ind in range(1,R):
                if t[loc]>t[ind]:
                    loc = ind

            # If we have reached our end time, break the script.
            if T[i] + t[loc] > self._T_s:
                count = i
                print(T[i] + t[loc])
                break

            # update the state of the system, and catalog the jump time.
            # update the integrated intensity functions.
            sol[:,i+1] = sol[:,i] + zeta[:,loc]
            T[i+1] = T[i] + t[loc]
            for c in range(0,R):
                Tk[c] = Tk[c] + lamb[c]*t[loc]

            # find the next jump time of the one unit Poisson process that fired.
            r = self._rng.uniform(0, 1)
            Pk[loc] = Pk[loc] + np.log(1/r)
        ####
        state = sol[:, count]
        state += self._rng.normal(0.0, self._system_noise)
        state = np.clip(state, 0.0, np.inf)
        self._trajectory.append(state)
        # observation
        observation = self._observe(state)
        self._observations.append(observation)
        # reward
        T = np.array([self._steps_done * self._T_s])
        reference = self.ref_trajectory(T)[0]
        reward = self._compute_reward(state[2], reference[0], reward_func)
        self._rewards.append(reward)
        # done
        done = False
        # info
        info = {
            'G': state[2],
            'tolerance': self._compute_reward(state[2], reference[0], 'tolerance'),
        }
        # step
        self._steps_done += 1
        # noise corrupted G (and t) observed
        if self._observation_mode == 'partially_observed':
            return observation, reward, done, info
        # perfect R, P, G (and t) observed
        return state, reward, done, info


# class DiscreteTimeCRN(ContinuousTimeDiscreteActionCRN):
#    """
#    s' = A @ s + B @ a
#    """
#
#    def __init__(
#        self,
#        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
#        sampling_rate: float = 10,
#    ) -> None:
#        super().__init__(ref_trajectory, sampling_rate)


# class MarginalParticleFilter():
#
#    # disturbance
#    d = 0.0
#    # parameters
#    theta = np.array([d_r, d_p, k_m, b_r, d])
#    # sample parameter particles
#    sigma_theta = 1e-3 * np.array([1.0155, 0.0509, 0.0150, 1.0347, 0.1000])
#    # perturb parameter particles
#    sigma_p = 1e-3 * np.array([0.2539, 0.0127, 0.0037, 0.2587, 0.0250])
#    # perturb state particles
#    sigma_s = np.array([0.0100, 0.0100, 0.0100])
#    # compute particle weights
#    sigma_meas = 0.0025
#
#    def __init__(
#        self,
#        rng: np.random.Generator,
#        P: int = 5,
#        sampling_rate: float = 10,
#    ) -> None:
#        self.rng = rng
#        self.P = P
#        self._T_s = sampling_rate
#        # initialize
#        self.param_p = None
#        self.state_p = None
#        self._setup_particles()
#        self.curr_step = 0
#
#    @property
#    def meas_p(self) -> np.ndarray:
#        if self.state_p is None:
#            return None
#        # C particles shape (P, 1, 3)
#        C_p = C.reshape(1, 1, -1).repeat(self.P, axis=0)
#        # y_n = C x_n
#        return C_p @ self.state_p
#
#    def __call__(self, action: float, meas: float):
#        self._propagate_state_p(action)
#        self._resample_particles(meas)
#        self._perturb_param_p()
#        self.curr_step += 1
#        return self.param_p, self.state_p
#
#    def _setup_particles(self) -> None:
#        # parameter particles shape (P, 5, 1)
#        self.param_p = self.theta.reshape(1, -1, 1).repeat(self.P, axis=0)
#        self.param_p += self._wn(self.sigma_theta)
#        # state particles shape (P, 3, 1)
#        self.state_p = np.ones((self.P, 3, 1))
#
#    def _propagate_state_p(self, action: float) -> None:
#        A_p = []
#        B_p = []
#        action_p = []
#        for _theta in self.param_p:
#            _A, _B = self._A_B(_theta, self._T_s)  # global T_s
#            A_p.append(_A)
#            B_p.append(_B)
#            action_p.append(self._a(_theta, action))
#        # A particles shape (P, 3, 3)
#        A_p = np.stack(A_p, axis=0)
#        # B particles shape (P, 3, 2)
#        B_p = np.stack(B_p, axis=0)
#        # action particles shape (P, 2, 1)
#        action_p = np.stack(action_p, axis=0)
#        # x_n+1 = A(theta_n) x_n + B(theta_n) a(u_n-1, theta_n) + w_n+1
#        self.state_p = A_p @ self.state_p + B_p @ action_p
#        self.state_p += self._wn(self.sigma_s)
#
#    def _resample_particles(self, meas: float) -> None:
#        # particle weights shape (P, 1, 1)
#        #w_p = np.exp(- (self.meas_p - meas) ** 2 / (2 * self.sigma_meas ** 2))
#        w_p = - np.log(np.sqrt(2 * np.pi) * self.sigma_meas) - (self.meas_p - meas) ** 2 / (2 * self.sigma_meas ** 2)
#        # normalize particle weights
#        W_p = w_p / w_p.sum()
#        # resample parameter particles
#        self.param_p *= W_p
#        # resample state particles
#        self.state_p *= W_p
#
#    def _perturb_param_p(self) -> None:
#         # theta_n+1 = theta_n + v_n+1
#         self.param_p += self._wn(self.sigma_p)
#
#    def _wn(self, sigma: np.ndarray) -> np.ndarray:
#        mu = np.zeros(sigma.shape)
#        cov = np.diag(sigma)
#        return self.rng.multivariate_normal(mu, cov, self.P).reshape(self.P, -1, 1)
#
#    @staticmethod
#    def _A_B(
#        _theta: np.ndarray,
#        _T_s: float,
#    ) -> typing.Tuple[np.ndarray, np.ndarray]:
#        _d_r, _d_p, _k_m, _b_r, _d = _theta.reshape(-1)
#        _A_c = np.array([[-_d_r, 0.0, 0.0],
#                         [_d_p + _k_m, -_d_p - _k_m, 0.0],
#                         [0.0, _d_p, -_d_p]])
#        _B_c = np.array([[_d_r, _b_r],
#                         [0.0, 0.0],
#                         [0.0, 0.0]])
#        _A = np.exp(_A_c * _T_s)
#        _B = np.linalg.inv(_A_c) @ (_A - np.eye(_A.shape[0])) @ _B_c
#        return _A, _B
#
#    @staticmethod
#    def _a(
#        _theta: np.ndarray,
#        action: float,
#    ) -> np.ndarray:
#        _d_r, _d_p, _k_m, _b_r, _d = _theta.reshape(-1)
#        return np.array([[1.0],
#                         [action + _d]])
