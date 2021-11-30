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
d_r = 0.0956
d_p = 0.0214
k_m = 0.0116
b_r = 0.0965

A_c = np.array([[-d_r, 0.0, 0.0],
                [d_p + k_m, -d_p - k_m, 0.0],
                [0.0, d_p, -d_p]])

B_c = np.array([[d_r, b_r],
                [0.0, 0.0],
                [0.0, 0.0]])

C = np.array([[0.0, 0.0, 1.0]])

# refer to https://static-content.springer.com/esm/art%3A10.1038%2Fncomms12546/MediaObjects/41467_2016_BFncomms12546_MOESM1324_ESM.pdf
# equivalent discrete-time fold-change model:
#     s' = A @ s + B @ a
T_s = 10  # experimental observation sampling rate

A = np.exp(A_c * T_s)

B = np.linalg.inv(A_c) @ (A - np.eye(A.shape[0])) @ B_c


class ContinuousTimeDiscreteActionCRN(Env):
    """
    ds / dt = A_c @ s + B_c @ a
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
    ) -> None:
        super().__init__()
        # reference trajectory generator
        self.ref_trajectory = ref_trajectory
        # sampling rate
        self._T_s = sampling_rate
        # initialize
        self._trajectory = []
        self._actions_taken = []
        self._rewards = []
        self._steps_done = 0

    @property
    def state(self) -> typing.Optional[typing.List[np.ndarray]]:
        if self._trajectory:
            return self._trajectory[self._steps_done]
        else:
            return None

    @property
    def state_dim(self) -> int:
        return A_c.shape[1]

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
        self._trajectory = []
        self._actions_taken = []
        self._rewards = []
        self._steps_done = 0
        state = np.ones((self.state_dim,))
        self._trajectory.append(state)
        return state

    @staticmethod
    def func(t: float, y: np.ndarray, action: float) -> np.ndarray:
        a = np.array([1.0, action])
        return A_c @ y + B_c @ a

    def step(self, action, mode: str):
        if self.state is None:
            raise RuntimeError
        if self.discrete:
            action = (action + 1) / self.action_dim  # float
        else:
            action = action[0]  # float
        action = np.clip(action + self._rng.normal(0, 1e-3), 0, 1)
        # environment dynamics simulation sampling rate
        delta = 0.1
        sol = solve_ivp(
            self.func,
            (0, self._T_s + delta),
            self.state,
            t_eval=np.arange(0, self._T_s + delta, delta),
            args=(action,),
        )
        state = sol.y[:, -1]
        self._trajectory.append(state)
        self._actions_taken.append(action)
        self._steps_done += 1
        observation = state[2]
        reference = self.ref_trajectory(np.array([self._steps_done * self._T_s]))[0][0]
        reward = self.compute_reward(observation, reference, mode)
        self._rewards.append(reward)
        done = False
        info = {}
        return state, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, mode: str):
        abs_dist = abs(desired_goal - achieved_goal)
        tolerance = self.ref_trajectory.tolerance
        perfection = tolerance * 0.01
        if mode == 'nega_abs':
            return (- abs_dist)
        elif mode == 'inverse_abs':
            if (abs_dist < perfection):
                return (1/perfection)
            else:
                return (1/abs_distance)
        elif mode == 'nega_logabs':
            return (-np.log(abs_dist))
        elif mode == 'expabs':
            return (np.exp(abs_dist))
        elif mode == 'percentage':
            return (1. - abs_dist / desired_goal)
        elif mode == 'tolerance':
            return 1 if (abs_dist / desired_goal < 2 * tolerance) else 0
        elif mode == 'graded_tolerance':
            return (1. - abs_dist / desired_goal) if (abs_dist / desired_goal < 2 * tolerance) else 0
        else:
            raise RuntimeError

    def render(
        self,
        mode: str = 'human',
        trajectory: typing.Optional[typing.List] = None,
        actions_taken: typing.Optional[typing.List] = None,
        steps_done: typing.Optional[int] = None
    ) -> None:
        if (self.state is None) and (not trajectory):
            raise RuntimeError
        # for replaying
        _trajectory = self._trajectory if trajectory is None else trajectory
        _actions_taken = self._actions_taken if actions_taken is None else actions_taken
        ## Add line here to retrieve stagewise reward
        _rewads = self._rewards
        _steps_done = self._steps_done if steps_done is None else steps_done
        # simulation sampling rate
        delta = 0.1
        # reference trajectory and tolerance margin
        t = np.arange(0, self._T_s * _steps_done + delta, delta)
        ref_trajectory, tolerance_margin = self.ref_trajectory(t)
        # sfGFP
        T = np.arange(0, self._T_s * _steps_done + self._T_s, self._T_s)
        R, P, G = np.stack(_trajectory, axis=1)
        # intensity
        t_u = np.concatenate([
           np.arange(self._T_s * i, self._T_s * (i + 1) + 1) for i in range(_steps_done)
        ])
        u = np.array(_actions_taken).repeat(self._T_s + 1) * 100
        # plot
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1, ## Change this to 2, to have the evolution of the stepwise reward on the side
            sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )
        axs[0].plot(t, ref_trajectory, '--', color='grey')
        axs[0].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
        if mode == 'human':
            axs[0].plot(T, G, 'o-', label='G', color='g')
        else:
            axs[0].plot(T, R, 'o-', label='R', color='r')
            axs[0].plot(T, P, 'o-', label='P', color='b')
            axs[0].plot(T, G, 'o-', label='G', color='g')
        axs[0].set_ylabel('sfGFP (1/min)')
        axs[0].legend()
        axs[1].plot(t_u, u)
        axs[1].set_xlabel('Time (min)')
        axs[1].set_ylabel('intensity (%)')
        ## Add an axis to show evolution of stepwise reward in 2nd column
        axs[2].plot(_rewads)
        axs[2].set_ylabel('rewads')
        plt.show()

    def close(self) -> None:
        self._trajectory = []
        self._actions_taken = []
        self._rewards = []
        self._steps_done = 0


class ContinuousTimeContinuousActionCRN(ContinuousTimeDiscreteActionCRN):
    """
    s' = A @ s + B @ a
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
    ) -> None:
        super().__init__(ref_trajectory, sampling_rate)

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
    ) -> None:
        super().__init__(ref_trajectory, sampling_rate)

    def step(self, action, mode: str):
            if self.state is None:
                raise RuntimeError
            if self.discrete:
                action = (action + 1) / self.action_dim  # float
            else:
                action = action[0]  # float
            ####
            # A maximum number of steps to run before breaking.
            maxi = T_s*10**5

            # Define all rate parameters in the model.
            # Each column is a reaction vector: \zeta_k = y_k' - y_k.
            # Total number of reaction channels in the model.
            k = np.array([b_r*action, b_p, d_r, d_p, d_p, k_m, k_m])
            zeta = np.array([[0,  0,  0,  0,  0, 0, 0],
                    [1,  0, -1,  0,  0,  0, 0],
                    [0,  1,  0, -1, 0,  -1, +1],
                    [0,  0,  0,  0,  -1, +1, -1]])
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
            Tk = np.zeros(R)
            Pk = np.zeros(R)
            t = np.zeros(R)

            # uniform random variables in order to find the first jump time of each unit Poisson process.
            # set first jump time of each unit Poisson process.
            r = np.random.rand(R)
            Pk = np.log(1./r)

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
                if T[i] + t[loc] > T_s:
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
                r = np.random.rand()
                Pk[loc] = Pk[loc] + np.log(1/r)
            ####
            state = sol[:,count]
            self._trajectory.append(state)
            self._actions_taken.append(action)
            self._steps_done += 1
            observation = state[3] # gene copy number added here vs. deterministic case
            reference = self.ref_trajectory(np.array([self._steps_done * self._T_s]))[0][0]
            reward = self.compute_reward(observation, reference, mode)
            done = False
            info = {}
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
