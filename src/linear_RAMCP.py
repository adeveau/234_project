import numpy as np
#import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Input, Dense
#from keras import optimizers
#import keras.backend as K

class StateNode(object):
    def __init__(self, tail, nS, nA, z, n_envs, depth = 0, parent = None):
        self.children = [None]*nA
        self.action_values = [0]*nA
        self.depth = depth
        self.count = 0
        self.counts = [0]*n_envs
        self.avg_action_cts = [0]*nA
        self.tail = tail
        self.z = z
        self.parent = parent

    def value(self):
        return max(self.action_values)

    def history(self):
        hist = []
        node = self
        while node.parent is not None:
            hist.append(node.tail)
            node = node.parent
        return hist[::-1]

class ActionNode(object):
    def __init__(self, nS, tail = None, parent = None):
        self.children = [None]*nS
        self.tail = tail
        self.parent = parent

    def history(self):
        hist = []
        node = self
        while node.parent is not None:
            hist.append(node.tail)
            node = node.parent
        return hist[::-1]

class linear_RAMCP(object):
    def __init__(self, prior, env, nS, nA, max_depth = 10, gamma = .9, s0 = 0, n_trans = 1, lr = .1):
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
        self.lr = lr

        init_z = np.ones(len(self.envs))/float(len(self.envs))
        self.root = StateNode(tail = s0, nS = self.nS, nA = self.nA, depth = 0, z = init_z, n_envs = len(self.envs))
        self.V  = [0]*len(self.envs)
        self.n_trans = n_trans

        self.weights = np.zeros(self.feature_vec(self.root, 0).shape)
        self.adv_history = []


    def estimateV(self, node, idx):
        if node.depth > self.max_depth:
            return 0

        node.count += 1
        node.counts[idx] += 1
        action_values = self.estimateQ(node, idx)
        
        #We sample every action every time, so we need to reweight based on b_adv_cur
        #Importance-sampling-esque
        #w = len(self.envs)*self.b_adv_cur[idx]

        #Update the action counts to track the average policy
        for action, action_value in enumerate(action_values):
            #print(self.weights, self.computeQ(node, action))
            self.weights += .1*(action_value - self.computeQ(node, action))*self.feature_vec(node, action)

        argmax = np.argmax([self.computeQ(node, a) for a in range(self.nA)])
        node.avg_action_cts[argmax] += 1

        return action_values[argmax]

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
                node_added = self.add_nodes(node, state, action, idx)

                #Recurse 
                next_node = node.children[action].children[state]


                for child in node.children[action].children:
                    if child is not None:
                        child.z[idx] = node.z[idx] * child.counts[idx]/float(node.counts[idx])

                v = self.estimateV(next_node, idx)
                new_values[action] += 1./(self.n_trans) * (reward + self.gamma*v)

        return new_values

    def add_nodes(self, node, state, action, idx):        
        new_z = np.zeros(len(self.envs))
        new_z[idx] = node.z[idx]/float(node.counts[idx]+1)

        if node.children[action] is None:
            node.children[action] = ActionNode(self.nS, parent = node, tail = action)
            node.children[action].children[state] =  StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1, z = new_z, n_envs = len(self.envs), parent = node.children[action])
            return True
        else:
            act = node.children[action]
            if act.children[state] is None:
                act.children[state] =  StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1, z = new_z, n_envs = len(self.envs), parent = act)
                return True
        return False

    def update_b_adv(self):
        """
        Solve a linear program to find an approximate best response
        """
        self.b_adv_cur = optimize.linprog(self.V, A_eq = np.ones((1, len(self.prior))), b_eq = np.array([1])).x
        self.b_adv_avg += (self.b_adv_cur - self.b_adv_avg)/self.n_iter

    def step(self):
        self.n_iter += 1
        self.adv_history.append(self.b_adv_avg.copy())
        idx = np.random.choice(range(len(self.envs)), p = self.b_adv_cur)
        r = self.estimateV(self.root, idx)
        for idx in range(len(self.envs)):
            self.V[idx] += (self.greedy_rollout(self.root, idx) - self.V[idx])/self.n_iter

        self.update_b_adv()
        if self.n_iter % 100 == 0:
            self.lr *= .99


    def greedy_rollout(self, node, idx):
        cur_env = self.envs[idx]
        cur_env.state = node.tail
        total_reward = 0
        discount = 1
        done = False
        while not done and node.depth < self.max_depth + 1:
            action = np.argmax([self.computeQ(node, a) for a in range(self.nA)])
            state, reward, done, _ = cur_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            self.add_nodes(node, state, action, idx)
            node = node.children[action].children[state]
        return total_reward

    def run(self, n):
        for x in range(n):
            self.step()


    def computeQ(self, node, action):
        return np.dot(self.weights, self.feature_vec(node, action))

    def feature_vec(self, node, action):
        return np.array([1] + list(node.z) + [action])
        f_vec = node.history() + [action]
        f_vec.extend([0]*(2*self.max_depth + 2 - len(f_vec)))
        return np.array(f_vec)

def walk(node, r):
    if isinstance(node, StateNode) and len(r.feature_vec(node, 0) < 9):
        print("action values: {}".format([r.computeQ(node, a) for a in (0,1)]))
        print( "z :{}".format(node.z))

    for c in node.children:
        if c is not None:
            walk(c, r)

