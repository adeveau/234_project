import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs import classic_control
import numpy as np

class ModPendulumEnv(classic_control.PendulumEnv):
    def __init__(self, g):
        self.g = g
        super(ModPendulumEnv, self).__init__()

    def step(self, u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
