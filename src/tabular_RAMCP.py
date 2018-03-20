import numpy as np
#import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
#from noisy_nchain import NoisyNChainEnv


class StateNode(object):
    def __init__(self, tail, nS, nA, n_env, depth = 0):
        self.children = [None]*nA
        self.action_values = [0]*nA
        self.depth = depth
        self.count = 0
        self.counts = [0]*n_env
        self.avg_action_cts = [0]*nA
        self.tail = tail

    def value(self):
        return max(self.action_values)

class ActionNode(object):
    def __init__(self, nS):
        self.children = [None]*nS

class tab_RAMCP(object):
    def __init__(self, prior, env, nS, nA, max_depth = 10, gamma = .9, s0 = 0, n_trans = 1, bootstrap = False):
        self.gamma = gamma
        self.nS = nS
        self.nA = nA

        self.prior = []
        self.envs = []
        for tup in prior: 
            self.prior.append(tup[1])
            self.envs.append(env(**tup[0]))

        self.prior = np.array(self.prior)
        self.b_adv_cur = self.prior.copy()
        self.b_adv_avg = self.prior.copy()
        self.max_depth = max_depth
        self.n_iter = 0

        self.root = StateNode(tail = s0, nS = self.nS, nA = self.nA, n_env = len(self.envs), depth = 0)
        self.V  = [0]*len(self.envs)
        self.n_trans = n_trans

        self.bootstrap = bootstrap
        self.adv_history = []

    def estimateV(self, node, idx):
        if node.depth > self.max_depth:
            return 0

        action_values = self.estimateQ(node, idx)
        node.count += 1
        node.counts[idx] += 1

        #Update action_values based on the new samples from estimateQ
        argmax = 0
        cur_max = -np.inf

        #We sample every action every time, so we need to reweight based on b_adv_cur
        #Importance-sampling-esque
        w = len(self.envs)*self.b_adv_cur[idx]

        for i, (old_val, new_val) in enumerate(zip(node.action_values, action_values)):
            #Weighted Monte Carlo update
            node.action_values[i] += (w*new_val - old_val)/(node.counts[idx])
            if old_val > cur_max:
                cur_max = old_val
                argmax = i           

        #Update the action counts to track the average policy
        node.avg_action_cts[argmax] += 1

        if not self.bootstrap:
            return action_values[argmax]
        else:
            return node.action_values[argmax]

    def estimateQ(self, node, idx):
        cur_env = self.envs[idx]
        new_values = [0]*self.nA
        for action in range(self.nA):
            for x in range(self.n_trans):
                #First reset the state so we can sample again
                cur_env.state = node.tail

                #Sample a new transition
                state, reward, done, _ = cur_env.step(action)

                #Add nodes to the tree if necessary
                self.add_nodes(node, state, action)


                #Recurse 
                v = self.estimateV(node.children[action].children[state], idx)
                new_values[action] += 1./(self.n_trans) * (reward + self.gamma*v)

        return new_values


    def update_b_adv(self):
        """
        Solve a linear program to find an approximate best response
        """
        self.b_adv_cur = optimize.linprog(self.V, A_eq = np.ones((1, len(self.prior))), b_eq = np.array([1])).x
        self.b_adv_avg += (self.b_adv_cur - self.b_adv_avg)/self.n_iter

    def step(self):
        self.n_iter += 1
        self.adv_history.append(self.b_adv_avg.copy())
        for idx in range(len(self.envs)):
            r = self.estimateV(self.root, idx) 
            self.V[idx] += (self.greedy_rollout(self.root, idx) - self.V[idx])/self.n_iter

        """
        idx = np.random.choice(range(len(self.envs)), p = self.b_adv_cur)
        r = self.estimateV(self.root, idx)
        self.V[idx] += (r - self.V[idx])/self.n_iter
        """   
        self.update_b_adv()

    def run(self, n):
        for x in range(n):
            self.step()

    def greedy_rollout(self, node, idx):
        cur_env = self.envs[idx]
        cur_env.state = node.tail
        total_reward = 0
        discount = 1
        done = False
        while not done and node.depth < self.max_depth + 1:
            action = np.argmax(node.action_values)
            state, reward, done, _ = cur_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            self.add_nodes(node, state, action)
            node = node.children[action].children[state]
        return total_reward

    def add_nodes(self, node, state, action):
        if node.children[action] is None:
            node.children[action] = ActionNode(self.nS)
            node.children[action].children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, n_env = len(self.envs), depth = node.depth + 1)
        else:
            act = node.children[action]
            if act.children[state] is None:
                act.children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, n_env = len(self.envs), depth = node.depth + 1)

def walk(node, a):
    for c in node.children:
        if c is not None:
            if isinstance(c, StateNode):
                a[0] += 1
                if not (c.children[0] is None and c.children[1] is None):
                    print(np.array(c.action_values).argmax())
            walk(c, a)

