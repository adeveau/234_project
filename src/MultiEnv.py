import numpy as np
from scipy import optimize
from gym import spaces

#Kinda awful, but not sure of any other way to make this work
#Dynamically inherits from the class of envs[0]
def make_multienv(envs, max_depth = np.inf, gamma = 1):

    def step(self, action):
        if not self.done:

            #Update what needs to be updated
            self.cur_depth += 1
            self.done = (self.cur_depth >= self.max_depth)
            state, reward, done, _ =  self.cur_env.step(action)
            self.history.extend([state, action, reward])
            padded_history = np.array(self.history + [0]*(3*self.max_depth - len(self.history)), dtype = np.float32)

            output = (padded_history, reward, self.done, _)
            self.total_reward += reward*gamma**(self.cur_depth)
            self.state = padded_history
            return output

        else:
            #if self.n_iter[self.idx] > 0:
            #    self.V[self.idx] += (self.total_reward - self.V[self.idx])/float(self.n_iter[self.idx])

            self.idx = np.random.choice(range(self.n_envs), p = self.b_adv_cur)
            self.cur_env = self.envs[self.idx]

            #Reset what needs to be reset
            self.history = []
            self.cur_env.reset()
            self.cur_depth = 1
            self.done = (self.cur_depth >= self.max_depth)
            self.n_iter[self.idx] += 1

            self.total_reward = 0
            self.update_b()

            return self.step(action)

    def reset(self):
        for env in self.envs:
            env.reset()
        #self.__init__(self.envs, self.max_depth)
        self.history = []
        self.state = np.array([0.]*self.max_depth*3)
        return self.state

    def hard_reset(self):
        self.reset()
        self.__init__(self.envs, self.max_depth)
        return self.state

    def update_b(self):
        self.b_adv_cur = optimize.linprog(self.V, A_eq = np.ones((1, self.n_envs)), b_eq = np.array([1])).x
        self.b_adv_avg += (self.b_adv_cur - self.b_adv_avg)/self.n_iter.sum()

    def greedy_rollout(self, Q, idx):
        cur_env = self.envs[idx]
        cur_env.reset()
        total_reward = 0
        done = False
        history = []
        cur_depth = 1
        while not done:
            if tuple(history) in Q:
                action = np.argmax(Q[tuple(history)])
            else:
                action = cur_env.action_space.sample()
            state, reward, done, _ = cur_env.step(action)
            cur_depth += 1
            done = done or (cur_depth >= max_depth)
            history.extend([state, action, reward])
            total_reward += reward
        return total_reward

    #Dynamically create a new class inheriting from the class of envs[0]
    #with the above attributes
    cls = envs[0].__class__
    new_multi_env = type("MultiEnv_{}".format(cls.__name__), 
                        (cls,), 
                        {'step' : step, 'update_b' : update_b, 'reset' : reset, 
                        'nS' : envs[0].observation_space.n, 'nA' : envs[0].action_space.n,
                        'hard_reset' : hard_reset, 'greedy_rollout' : greedy_rollout})


    def __init__(self, envs, max_depth = np.inf, gamma = 1):
        super(new_multi_env, self).__init__()
        self.envs = envs
        self.n_envs = len(self.envs)
        self.done = True

        self.cur_depth = 0
        self.b_adv_cur = np.ones(self.n_envs)/float(self.n_envs)
        self.b_adv_avg = self.b_adv_cur


        self.cur_env = None


        self.epsilon = .1/self.n_envs
        self.max_depth = max_depth

        self.n_iter = np.zeros(self.n_envs, dtype = np.int32)
        self.history = []

        self.total_reward = 0
        self.gamma = gamma
        self.V = np.zeros(self.n_envs)
        self.state = np.array([0]*self.max_depth*3)

        self.idx = 0
        self.observation_space = spaces.Box(-10, 10, shape = (self.max_depth*3, ))


    #Have to add __init__ separately to get the call to super to work :/
    new_multi_env.__init__ = __init__

    #Initialize and return
    return new_multi_env(envs, max_depth, gamma)

"""
class MultiEnv(gym.env):
    def __init__(self, envs, max_depth = None):
        self.envs = envs
        self.n_evns = len(self.envs)
        self.done = False
        self.cur_depth = 0
        self.b = np.ones(self.n_envs)/float(self.n_envs)
        self.cur_env = 0
        self.epsilon = .1/self.n_envs

    def step(action):
        if not done:
            return self.cur_env.step(action)
        else:
            #We add a small self.epsilon to each entry in self.b to ensure non-zero probability of selecting
            #each environment. To ensure convergence, we must have self.epsilon -> 0 as n -> infinity
            cur_env = self.envs[np.random.choice(range(self.n_envs), p = (self.b + self.epsilon)/(1 + self.n_envs*self.epsilon)]
            done = False
            step(action)
"""