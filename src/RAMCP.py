import numpy as np
import mod_pend.envs as envs
from scipy import optimize
from gym.envs import toy_text
import time
import copy

class StateNode(object):
    def __init__(self, tail, depth = 0):
        self.children = {}
        self.action_values =  {}
        self.depth = depth
        self.count = 0
        self.avg_action_cts = {}
        self.tail = tail

    def value(self):
        return max(self.action_values)

class ActionNode(object):
    def __init__(self):
        self.children = {}

class RAMCP(object):
    def __init__(self, prior, env, max_depth = 10, gamma = .9, s0 = None, n_trans = 1):
        self.gamma = gamma

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

        if s0 is None:
            s0 = np.zeros(2)

        self.root = StateNode(tail = s0, depth = 0)
        self.V  = [0]*len(self.envs)
        self.n_trans = n_trans

    def estimateV(self, node, idx, tol = .045):
        if node.depth > self.max_depth:
            return 0

        action_values = self.estimateQ(node, idx)
        node.count += 1

        #Update action_values based on the new samples from estimateQ

        #We sample every transition model every time, so we need to reweight based on b_adv_cur
        #Importance-sampling-esque
        w = len(self.envs)*self.b_adv_cur[idx]

        for i, (old_val, new_val) in enumerate(zip(node.action_values, action_values)):
            node.action_values[i] += (w*new_val - old_val)/(node.count)        

        #Set the argmax and update the action counts to track the average policy
        cur_max = -np.inf
        for action in node.action_values:
            if node.action_values[action] > cur_max:
                argmax = action 
                cur_max = node.action_values[action]

        node.avg_action_cts[argmax] += 1

        return action_values[argmax]

    def estimateQ(self, node, idx):
        cur_env = self.envs[idx]
        new_values = {}
        for y in xrange(10):
            for x in xrange(self.n_trans):
                #First reset the state so we can sample again
                cur_env.state = node.tail
                action = cur_env.action_space.sample()
                #Sample a new transition
                state, reward, done, _ = cur_env.step(action)

                #Add nodes to the tree if necessary
                if self.n_iter % 5 == 0 or self.n_iter == 1:
                    self.add_nodes(node, state, action)
                    update_action = tuple(action)
                    next_node = node.children[tuple(action)].children[tuple(obs_to_state(state))]
                else:
                    closest_action = closest(action, node.children)
                    closest_state = closest(state, node.children[closest_action])
                    next_node = node.children[closest_action].children[closest_state]
                    update_action = closest_action

                #Recurse 
                v = self.estimateV(next_node, idx)
                if tuple(action) in new_values:
                    new_values[update_action] += 1./(self.n_trans) * (reward + self.gamma*v)
                else:
                    new_values[update_action] = 1./(self.n_trans) * (reward + self.gamma*v)

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

    def add_nodes(self, node, state, action):
        state = tuple(obs_to_state(state))
        action = tuple(action)
        if action not in node.children:
            node.children[action] = ActionNode()
            node.children[action].children[state] = StateNode(tail = np.array(state), depth = node.depth + 1)
        else:
            act = node.children[action]
            if state not in node.children[action].children:
                act.children[state] = StateNode(tail = np.array(state), depth = node.depth + 1) 


def obs_to_state(obs):
    return np.array([np.arccos(obs[0]), obs[2]])

def closest(target, array):
    min_dist = np.inf
    for value in array:
        value = np.array(value)
        if np.norm(target - value) < min_dist:
            min_dist = np.norm(target - value)
            argmin = value

    return argmin

def walk(node, a):
    for c in node.children:
        if c is not None:
            if isinstance(c, StateNode):
                a[0] += 1
                if not (c.children[0] is None and c.children[1] is None):
                    print np.array(c.action_values).argmax()
            walk(c, a)

if __name__ == "__main__":
    np.set_printoptions(precision = 4)
    r = RAMCP([({'g' : 10}, 1./3), ({'g' : 5}, 2./3)], envs.ModPendulumEnv, n_trans = 1, max_depth = 10, gamma = 1)
    st = time.time()
    r.run(100)
    print "Runtime: {}".format(time.time() - st)
    print "V: {}".format(r.V)
    print "Adversarial distribution: {}".format(r.b_adv_avg)
    print "root values {}".format(r.root.action_values)
    #a = [0]
    #walk(r.root, a)

