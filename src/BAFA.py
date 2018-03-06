import numpy as np
#import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy

class StateNode(object):
    def __init__(self, tail, nS, nA, depth = 0):
        self.children = [None]*nA
        self.action_values = [0]*nA
        self.depth = depth
        self.count = 0
        self.avg_action_cts = [0]*nA
        self.tail = tail

    def value(self):
        return max(self.action_values)

class ActionNode(object):
    def __init__(self, nS):
        self.children = [None]*nS

class BAFA(object):
    def __init__(self, prior, env, nS, nA, max_depth = 10, gamma = .9, s0 = 0, epsilon = .1):
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

        self.root = StateNode(tail = s0, nS = self.nS, nA = self.nA, depth = 0)
        self.V  = [0]*len(self.envs)
        self.epsilon = epsilon

    def search(self, node, max_iter):
        for x in xrange(max_iter):
            self.n_iter += 1
            idx = np.random.choice(range(len(self.b_adv_cur)), p = self.b_adv_cur, size = 1)
            r = self.simulate(self, node, idx)
            self.V[idx] += (r - self.V[idx])/self.n_iter
            self.update_b_adv()

    def simulate(self, node, idx):
        if node.depth > self.max_depth:
            return 0

        if np.random.random() < self.epsilon:
            cur_action = np.random.choice(range(self.nA))
        else:
            cur_action = np.argmax(node.action_values)

        cur_env = self.envs[idx]

        cur_env.state = node.tail

        #Sample a new transition
        state, reward, done, _ = cur_env.step(action)    

        if node.children[action] is None:
            node.children[action] = ActionNode(self.nS)
            node.children[action].children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1)
        else:
            act = node.children[action]
            if act.children[state] is None:
                act.children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1)    

        R = reward + self.gamma*self.simulate(node.children[action].children[state], idx)

        ##Using one-hot encoding x(s,a)_{s', a'} from HW2
        node.action_values[cur_action]= node.action_values[cur_action] - self.lr*(node.action_values[cur_action] - R)
        return R

    def update_b_adv(self):
        self.b_adv_cur = optimize.linprog(self.V, A_eq = np.ones((1, len(self.prior))), b_eq = np.array([1])).x
        self.b_adv_cur[-1] = 1. - self.b_adv_cur[:-1].sum()
        self.b_adv_avg += (self.b_adv_cur - self.b_adv_avg)/self.n_iter 

    def run(self, max_iter):
        self.search(self.root, max_iter)

if __name__ == "__main__":
    print "working"
    r = BAFA([({'slip' : 1}, 1./2), ({'slip' : 0}, 1./1)], toy_text.NChainEnv, nS = 5, nA = 2, max_depth = 4, gamma = 1)
    st = time.time()
    r.run(10)
    print "Runtime: {}".format(time.time() - st)
    print "V: {}".format(r.V)
    print "Adversarial distribution: {}".format(r.b_adv_avg)
    print "root values {}".format(r.root.action_values)