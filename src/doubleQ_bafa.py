import numpy as np
#import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy

class StateNode(object):
    def __init__(self, tail, nS, nA, depth = 0):
        self.children = [None]*nA
        self.action_values = np.zeros((2, nA))
        self.depth = depth
        self.count = 0
        self.avg_action_cts = [0]*nA
        self.tail = tail

    def value(self, which_q):
        return max(self.action_values[which_q, :])

    def greedy_action(self, which_q = None):
        if which_q is None:
            return np.argmax(self.action_values.mean(0))

        return np.argmax(self.action_values[which_q, :])

class ActionNode(object):
    def __init__(self, nS):
        self.children = [None]*nS

class DoubleQBAFA(object):
    def __init__(self, prior, env, nS, nA, max_depth = 10, gamma = .9, s0 = 0, epsilon = .1, lr = .01):
        self.gamma = gamma
        self.nS = nS
        self.nA = nA
        self.lr = lr

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
            #idx = np.random.choice(range(len(self.b_adv_cur)), p = self.b_adv_cur, size = 1)[0]
            for idx in xrange(len(self.envs)):
                r = self.simulate(node, idx)
                self.V[idx] += (self.greedy_rollout(self.root, idx) - self.V[idx])/float(self.n_iter)
            self.update_b_adv()
            if self.n_iter % 100 == 0:
                self.epsilon *= .99

    def simulate(self, node, idx):
        if node.depth > self.max_depth:
            return 0



        if np.random.random() < self.epsilon:
            action = np.random.choice(range(self.nA))
        else:
            action = node.greedy_action()

        cur_env = self.envs[idx]

        cur_env.state = node.tail

        #Sample a new transition
        state, reward, done, _ = cur_env.step(action)    

        w = len(self.envs)*self.b_adv_cur[idx]

        self.add_nodes(node, state, action)

        next_node = node.children[action].children[state]
        which_q = np.random.randint(0,2)
        R = reward + self.gamma*next_node.action_values[int (not which_q), next_node.greedy_action(which_q)]
        node.action_values[int (not which_q), action] -=  w*max(.01,self.lr/self.n_iter)*(node.action_values[int (not which_q), action] - R)
        self.simulate(next_node, idx)

        ##Using one-hot encoding x(s,a)_{s', a'} from HW2

        return R

    def update_b_adv(self):
        self.b_adv_cur = optimize.linprog(self.V, A_eq = np.ones((1, len(self.prior))), b_eq = np.array([1])).x
        self.b_adv_avg += (self.b_adv_cur - self.b_adv_avg)/self.n_iter 

    def run(self, max_iter):
        self.search(self.root, max_iter)

    def greedy_rollout(self, node, idx):
        cur_env = self.envs[idx]
        cur_env.state = node.tail
        total_reward = 0
        discount = 1
        done = False
        while not done and node.depth < self.max_depth + 1:
            action = node.greedy_action()
            state, reward, done, _ = cur_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            self.add_nodes(node, state, action)
            node = node.children[action].children[state]
        return total_reward

    def add_nodes(self, node, state, action):
        if node.children[action] is None:
            node.children[action] = ActionNode(self.nS)
            node.children[action].children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1)
        else:
            act = node.children[action]
            if act.children[state] is None:
                act.children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1) 

if __name__ == "__main__":
    np.set_printoptions(precision = 4)
    r = DoubleQBAFA([({'slip' : 1}, 1./2), ({'slip' : 0}, 1./2)], 
                toy_text.NChainEnv, nS = 5, nA = 2,
                 max_depth = 1, gamma = 1, lr = .5, epsilon = .1)
    st = time.time()
    r.run(100000)
    print "Runtime: {}".format(time.time() - st)
    print "V: {}".format(r.V)
    print "Adversarial distribution: {}".format(r.b_adv_avg)
    print "root values {}".format(r.root.action_values)
