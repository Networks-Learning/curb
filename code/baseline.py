import numpy as np
from numpy import random
from os import listdir
from datetime import datetime
import time


class SimulationUserBase():
    def intensity(self, t, hist_skip_last=False, skip_last_share=False, skip_base=False):
        if skip_base:
            res = 0.0
        else:
            res = self.mu
        if self.history is not None:
            if hist_skip_last:
                if self.history.shape[0] > 1:
                    hist = np.copy(self.history[:-1, :])
                    shared_events = hist[hist[:, 2] == 1, :]
                else:
                    return res
            else:
                if skip_last_share:
                    hist = np.copy(self.history)
                    hist[-1, 2] = 0
                    shared_events = hist[hist[:, 2] == 1, :]
                else:
                    shared_events = self.history[self.history[:, 2] == 1, :]
            res += (self.kernel(t - shared_events[:, 1]) * self.a).sum()
        return res

    def reset(self):
        self.history = None
        self.t = 0


class SimulationUserFile(SimulationUserBase):

    def __init__(self, a, w, data, p_f):
        self.a = a
        self.mu = 0
        self.kernel = lambda t: np.exp(-t * w)
        self.data = data
        self.p_f = p_f
        data = np.array(data)
        self.number_events = data.shape[0]
        self.T = data[-1, 1] - data[0, 1]
        self.t = 0
        self.f = random.binomial(1, self.p_f, size=data.shape[0])
        del data

    def sample(self):
        data = self.data
        data = np.array(data)
        data[:, 1] = data[:, 1] - data[0, 1]
        for i in range(data.shape[0]):
            u_id, self.t, s = data[i, :]
            f = self.f[i]
            if self.history is None:
                self.history = np.array([[u_id, self.t, s, f, -1]])
            else:
                self.history = np.append(self.history, np.array(
                    [[u_id, self.t, s, f, -1]]), axis=0)

            yield [u_id, self.t, s, f, -1]
        self.t = self.T
        return None


class Simulation():
    def __init__(self, policy, user_activity):
        self.policy = policy
        self.user_activity = user_activity
        self.T = self.user_activity.T

    def reset(self):
        self.user_activity.reset()
        self.policy.reset()
        self.user_activity_sampler = self.user_activity.sample()

    def run(self, cut_on_tau=False):
        self.reset()
        tau = self.policy.sample()
        termination = False
        #print("Current tau:", tau)
        res = next(self.user_activity_sampler, None)
        if res is None:
            raise Exception("No User Activity generated")
        else:
            u_id, u_t, u_s, u_f, post = res
        if u_t > tau:
            termination = True

        while(True):
            if not termination:
                if u_f == 0:
                    #print(self.policy.intensity(tau, hist_skip_last=True),self.policy.intensity(tau, skip_last_share=True))
                    if random.uniform() * self.policy.intensity(tau, hist_skip_last=True) > self.policy.intensity(tau, skip_last_share=True):
                        tau = self.policy.sample(skip_last_share=True)

                    if u_s == 1:
                        tau_ = self.policy.sampleDiffShare(u_t, tau)
                        if tau_ < tau:
                            tau = tau_
                            self.policy.t = tau
                            #print("share no flag", tau, self.user_activity.history[:, 3].sum())
            res = next(self.user_activity_sampler, None)
            if res is None:
                return (self.user_activity.history if not cut_on_tau else self.user_activity.history[self.user_activity.history[:, 1] < tau, :],
                        None if tau == self.T else tau)
            else:
                u_id, u_t, u_s, u_f, post = res
            if u_t > tau:
                termination = True

        return (self.user_activity.history[:-1, :] if not cut_on_tau else self.user_activity.history[self.user_activity.history[:, 1] < tau, :],
                None if tau == self.T else tau)


class ExposurePolicy():

    def __init__(self, T, q, activity_model):
        self.T = T
        self.q = q
        self.activity_model = activity_model

    def reset(self):
        self.t = 0

    def intensity(self, t, hist_skip_last=False, skip_last_share=False):
        return (1.0 / np.sqrt(self.q)) * \
            self.activity_model.intensity(t, hist_skip_last=hist_skip_last,
                                          skip_last_share=skip_last_share, skip_base=True)

    def sample(self, skip_last_share=False):
        while(True):
            sum_u = self.intensity(self.t, skip_last_share=skip_last_share)
            # If the intensity is zero return a large time.
            if sum_u < np.finfo(float).eps:
                return self.T + np.finfo(float).eps
            t_ = random.exponential(1.0 / sum_u)
            self.t = t_ + self.t
            if self.t > self.T:
                self.t = self.T
                return self.T
            new_u = self.intensity(self.t, skip_last_share=skip_last_share)
            if random.uniform() < new_u / sum_u:
                return self.t

    def sampleDiffShare(self, t, tau):
        while(True):
            sum_u = self.intensity(
                t, skip_last_share=False) - self.intensity(t, skip_last_share=True)
            if sum_u < np.finfo(float).eps:
                return self.T + np.finfo(float).eps
            t_ = random.exponential(1.0 / sum_u)
            t = t_ + t
            if t > tau:
                return tau
            new_u = self.intensity(
                t, skip_last_share=False) - self.intensity(t, skip_last_share=True)

            if random.uniform() * sum_u < new_u:
                return t
