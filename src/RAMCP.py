import numpy as np
import mod_pend.envs as envs
from scipy import optimize

class StateNode(object):
    def __init__(self, tail, env, depth = 0):
        self.children = [None]*env.nS
        self.action_values = [0]*env.nA
        self.depth = depth
        self.count = 0
        self.avg_action_cts = [0]*env.nA
        self.tail = tail

    def value(self):
        return max(self.action_values)

class ActionNode(object):
    def __init__(self, env):
        self.children = [None]*env.nS

class RAMCP(object):
    def __init__(self, prior, env, max_depth, gamma, s0, n_trans = 1):
        self.gamma = gamma

        self.prior = []
        self.envs = []
        for k in prior: 
            self.prior.append(prior[k])
            self.envs.append(self.env(k))

        self.max_depth = max_depth
        self.b_adv_avg = self.prior
        self.b_adv_cur = self.prior
        self.n_iter = 0


        self.root = StateNode(tail = s0, self.envs[0], depth = 0)
        self.V  = [0]*len(self.envs)
        self.n_trans = n_trans

    def estimateV(self, node, idx):
        if node.depth > self.max_depth:
            return 0
        action_values = self.estimateQ(node, idx)
        self.count += 1

        argmax = 0
        cur_max = -np.inf
        for i, old_val, new_val in enumerate(zip(node.action_values, action_values)):
            w = self.b_adv_cur[i]*len(self.envs)

            node.action_values[i] = (w*new_val - old_val)/self.count
            if old_val > cur_max:
                cur_max = old_val
                argmax = i           

        node.avg_action_cts[argmax] += 1

        return cur_max

    def estimateQ(self, node, idx):
        cur_env = self.envs[idx]
        new_values = [0]*len(cur_env.action_space)
        for action in cur_env.action_space:
            for x in xrange(self.n_trans):
                cur_state = node.tail
                state, reward, done, _ = cur_env.step(action)
                if node.children[action] is None:
                    node.children[action] = ActionNode()
                    node.children[action].children[state] = StateNode(tail = state, env = cur_env, depth = node.depth + 1)
                else:
                    act = node.children[action]
                    if act.children[state] is None:
                        act.children[state] = StateNode(tail = state, env = cur_env, depth = node.depth + 1)

                v = self.estimateV(node.children[action].children[state], idx)
                new_values[action] += 1./self.n_trans * (reward + self.gamma*v)

        return new_values


    def update_b_adv(self):
        self.cur_b_adv = optimize.linprog(self.V, A_eq = self.prior, b_eq = np.array([1]))
        self.avg_b_adv += (self.cur_b_adv - self.avg_b_adv)/self.n_iter

    def step(self):
        for idx in xrange(len(self.envs)):
            r = self.estimateV(self.root, idx) 
            self.V[idx] += (r - self.V[idx])/self.n_iter          
            self.n_iter += 1

    def run(self, n):
        for x in xrange(n):
            self.step()

if __name__ == "__main__":
    r = RAMCP({.5 : {'g': 10}, .5 :{'g' : 8}}, envs.ModPendulumEnv)
    r.run(5)
