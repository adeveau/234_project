from MultiEnv import make_multienv
from baselines import deepq
from gym.envs import toy_text

model = deepq.models.mlp([64])
env = make_multienv([toy_text.NChainEnv(slip = x) for x in (0,1)], 10)

act = deepq.learn(env, q_func = model, max_timesteps=10000, buffer_size = 10)


print("done")