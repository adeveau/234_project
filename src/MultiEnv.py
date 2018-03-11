from gym import envs
import numpy as np

#Kinda awful, but not sure of any other way to make this work
#Dynamically inherits from the class of envs[0]
def make_multienv(envs, max_depth = np.inf):
	cls = envs[0].__class__



	def step(self, action):
		if not self.done:
			self.cur_depth += 1
			self.done = (self.cur_depth >= self.max_depth)
			output =  self.cur_env.step(action)
			output = (output[0], output[1], self.done, output[3])
			return output
		else:
			#We add a small value self.epsilon to each entry in self.b to ensure non-zero probability of selecting
			#each environment. To ensure convergence, we must have self.epsilon -> 0 as n -> infinity
			self.cur_env = self.envs[np.random.choice(range(self.n_envs), p = (self.b + self.epsilon)/(1 + self.n_envs*self.epsilon))]
			self.cur_env.reset()
			self.cur_depth = 1
			self.done = (self.cur_depth >= self.max_depth)
			self.step(action)

	def reset(self):
		for env in self.envs:
			env.reset()
		self.cur_env.reset()

	new_multi_env = type("MultiEnv_{}".format(cls.__name__), 
						(cls,), 
						{'step' : step}) 

	def __init__(self, envs, max_depth = np.inf):
		super(new_multi_env, self).__init__()
		self.envs = envs
		self.n_envs = len(self.envs)
		self.done = True
		self.cur_depth = 0
		self.b = np.ones(self.n_envs)/float(self.n_envs)
		self.cur_env = None
		self.epsilon = .1/self.n_envs
		self.max_depth = max_depth

	new_multi_env.__init__ = __init__

	return new_multi_env(envs, max_depth)

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