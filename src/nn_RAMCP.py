import numpy as np
#import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy
from keras.models import Sequential
from keras.layers import Input, Dense
from keras import optimizers
import keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class StateNode(object):
    def __init__(self, tail, nS, nA, z, n_envs, depth = 0):
        self.children = [None]*nA
#        self.action_values = [0]*nA
        self.depth = depth
        self.count = 0
        self.counts = [0]*n_envs
        self.avg_action_cts = [0]*nA
        self.tail = tail
        self.z = z

    def value(self):
        return max(self.action_values)

class ActionNode(object):
    def __init__(self, nS):
        self.children = [None]*nS

class nn_RAMCP(object):
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

        self.model = self.build_model()
        self.adv_history = []
        
    def estimateV(self, node, idx):
        if node.depth > self.max_depth:
            return 0

        node.count += 1
        node.counts[idx] += 1
        action_values = self.estimateQ(node, idx)

        r = np.random.random()
        argmax = np.argmax(self.computeQ(node))
        
        #We sample every action every time, so we need to reweight based on b_adv_cur
        #Importance-sampling-esque
        w = len(self.envs)*self.b_adv_cur[idx]

        self.model.fit(x = feature_vec(node), y = w*action_values, verbose = 0)

        #Update the action counts to track the average policy
        

        #if r < .5:
        #    argmax = np.argmax(self.computeQ(node))
        node.avg_action_cts[argmax] += 1
        return action_values[0, argmax]

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

        return np.array(new_values).reshape((1,2))

    def add_nodes(self, node, state, action, idx):        
        new_z = np.zeros(len(self.envs))
        new_z[idx] = node.z[idx]/float(node.counts[idx])
        next_state_node = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1, z = new_z, n_envs = len(self.envs))
        if node.children[action] is None:
            node.children[action] = ActionNode(self.nS)
            node.children[action].children[state] = next_state_node
            return True
        else:
            act = node.children[action]
            if act.children[state] is None:
                act.children[state] = next_state_node
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
        for idx in range(len(self.envs)):
            r = self.estimateV(self.root, idx)
            self.V[idx] += (r - self.V[idx])/self.n_iter
        self.update_b_adv()
        if self.n_iter % 100 == 0:
            self.lr *= .99

    def run(self, n):
        for x in range(n):
            print(x)
            self.step()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim = len(self.envs), activation = 'relu'))
        model.add(Dense(self.nA, input_dim = 32, activation = 'linear'))

        sgd = optimizers.SGD(lr = .01, decay = 0, momentum = 0, nesterov = False)
        model.compile(loss = 'mean_squared_error', optimizer = sgd)

        return model

    def computeQ(self, node):
        return self.model.predict(feature_vec(node))

    def greedy_rollout(self, node, idx):
        cur_env = self.envs[idx]
        cur_env.state = node.tail
        total_reward = 0
        discount = 1
        done = False
        while not done and node.depth < self.max_depth + 1:
            action = np.argmax(self.computeQ(node))
            state, reward, done, _ = cur_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            self.add_nodes(node, state, action, idx)
            node = node.children[action].children[state]
        return total_reward

def feature_vec(node):
    return node.z.reshape((1,len(node.z)))

def walk(node, r):
    if isinstance(node, StateNode):
        print("action values: {}".format(r.computeQ(node)))
        print( "z :{}".format(node.z))

    for c in node.children:
        if c is not None:
            walk(c, r)
