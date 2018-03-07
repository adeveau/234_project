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

class RAMCP(object):
    def __init__(self, prior, env, nS, nA, max_depth = 10, gamma = .9, s0 = 0, n_trans = 1):
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
        self.n_trans = n_trans

    def estimateV(self, node, idx):
        if node.depth > self.max_depth:
            return 0

        action_values = self.estimateQ(node, idx)
        node.count += 1

        #Update action_values based on the new samples from estimateQ
        argmax = 0
        cur_max = -np.inf

        #We sample every action every time, so we need to reweight based on b_adv_cur
        #Importance-sampling-esque
        w = len(self.envs)*self.b_adv_cur[idx]

        for i, (old_val, new_val) in enumerate(zip(node.action_values, action_values)):
            #Weighted Monte Carlo update
            node.action_values[i] += (w*new_val - old_val)/(node.count)
            if old_val > cur_max:
                cur_max = old_val
                argmax = i           

        #Update the action counts to track the average policy
        node.avg_action_cts[argmax] += 1

        return action_values[argmax]

    def estimateQ(self, node, idx):
        cur_env = self.envs[idx]
        new_values = [0]*self.nA
        for action in xrange(self.nA):
            for x in xrange(self.n_trans):
                #First reset the state so we can sample again
                cur_env.state = node.tail

                #Sample a new transition
                state, reward, done, _ = cur_env.step(action)

                #Add nodes to the tree if necessary
                if node.children[action] is None:
                    node.children[action] = ActionNode(self.nS)
                    node.children[action].children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1)
                else:
                    act = node.children[action]
                    if act.children[state] is None:
                        act.children[state] = StateNode(tail = state, nS = self.nS, nA = self.nA, depth = node.depth + 1)

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
        for idx in xrange(len(self.envs)):
            r = self.estimateV(self.root, idx) 
            self.V[idx] += (r - self.V[idx])/self.n_iter
        self.update_b_adv()

    def run(self, n):
        for x in xrange(n):
            self.step()

def walk(node, a):
    for c in node.children:
        if c is not None:
            if isinstance(c, StateNode):
                a[0] += 1
                print np.array(c.action_values).argmax()
            walk(c, a)

if __name__ == "__main__":
    r = RAMCP([({'slip' : 1}, 1./2), ({'slip' : 0}, 1./1)], toy_text.NChainEnv, 5, 2, n_trans = 2, max_depth = 3, gamma = 1)
    st = time.time()
    r.run(5000)
    print "Runtime: {}".format(time.time() - st)
    print "V: {}".format(r.V)
    print "Adversarial distribution: {}".format(r.b_adv_avg)
    print "root values {}".format(r.root.action_values)
    #a = [0]
    #walk(r.root, a)

