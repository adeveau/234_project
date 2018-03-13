from MultiEnv import make_multienv
from baselines import deepq
from gym.envs import toy_text, classic_control
from mod_pend.envs import *

model = deepq.models.mlp([64])
e = make_multienv([ModPendulumEnv(g = 10)], 10)
#e = classic_control.PendulumEnv()
e.reset()

act = deepq.learn(e, q_func = model, max_timesteps=100000, buffer_size = 1000)

print("V: {}".format(e.V))
print("Adversary: {}".format(e.b_adv_avg))
act.save("model.mdl")
print("done")