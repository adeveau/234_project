from nn_RAMCP import nn_RAMCP 
from linear_RAMCP import linear_RAMCP 
from tabular_RAMCP import tab_RAMCP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from gym.envs import toy_text
#import seaborn as sns

#sns.set_style("darkgrid")

if __name__ == "__main__":
    env2 = [({'slip' : .1}, .8), ({'slip' : .2}, .1), ({'slip' : .3}, .1)]
    env1 = [({'slip' : 1}, .8), ({'slip' : 0}, .2)]

    nn = nn_RAMCP(env2, toy_text.NChainEnv, 5, 2, n_trans = 1, max_depth = 3, gamma = 1)
    lin = linear_RAMCP(env2, toy_text.NChainEnv, 5, 2, n_trans = 1, max_depth = 3, gamma = 1)
    tab = tab_RAMCP(env2, toy_text.NChainEnv, 5, 2, n_trans = 1, max_depth = 3, gamma = 1, bootstrap = False)

    print("built")
    n = 500
    tab.run(n)
    print("tab")
    lin.run(n)
    print("lin")
    nn.run(n)
    print("nn")
    fig, ax = plt.subplots()

    ax.plot(range(n), [x[2] for x in nn.adv_history], label = "Neural Network")
    ax.plot(range(n), [x[2] for x in lin.adv_history], label = "Linear")
    ax.plot(range(n), [x[2] for x in tab.adv_history], label = "Tabular")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel(r"$p(\Theta_3)$")
    ax.legend()

    fig.savefig("three_envs_convergence3.png")
