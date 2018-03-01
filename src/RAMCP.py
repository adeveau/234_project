import numpy as np
from mod_pend import ModPendulumEnv

class RAMCP(object):
    def __init__(self, prior, env):
        self.prior = prior
        self.b_adv_avg = prior
        self.b_adv_cur = prior
        self.n_iter = 0
        self.V = np.zeros(len(prior))
        self.env = env

    def make_envs(self):
        self.envs = []
        for k in self.prior:
            self.envs.append(self.env(k))
   
    def estimateV(self, h, theta):
        pass

    def estimateQ(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    r = RAMCP({.5 : {'g': 10}, .5 :{'g' : 8}}, ModPendulumEnv)
    r.make_envs()
