import numpy as np
import gym
import time
from lake_envs import *
import copy
#import matplotlib.pyplot as plt

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method.
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  o_state = tuple(env.reset())
  e_orig = e
  q_alt = {o_state : np.zeros(env.nA)}
  avg_reward = 0
  rewards = np.zeros(num_episodes)
  avg_rewards = np.zeros(num_episodes)
  action_cts = [0,0]
  greedy_cts = [0,0]
  for x in range(num_episodes):
    old_state = tuple(env.reset())
    done = False
    episode_reward = 0
    while not done:
      rand = np.random.random()

      if rand > e:
        action = np.argmax(q_alt[old_state])

      else:
        action = env.action_space.sample()

      greedy_cts[np.argmax(q_alt[old_state])] += 1

      state, reward, done, _ = env.step(action)
      state = tuple(state)

      if state not in q_alt:
        q_alt[state] = np.zeros(env.nA)

      action_cts[action] += 1

      if not done:
        q_alt[old_state][action] = q_alt[old_state][action] + lr*(reward + gamma*q_alt[state].max() - q_alt[old_state][action])
      else:
        q_alt[old_state][action] = q_alt[old_state][action] + lr*(reward - q_alt[old_state][action])
        for x in range(env.n_envs):
          env.V[x] += (env.greedy_rollout(q_alt, x) - env.V[x])/float(env.n_iter.mean())

      old_state = state
      episode_reward += reward

    #q[old_state] = np.ones(env.nA)*reward
    #e = e_orig/(x)
    lr = lr/x
  print(e)
  return q_alt, avg_rewards, action_cts, greedy_cts

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters 
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print("Episode reward: %f" % episode_reward)

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q, avg_rewards = learn_Q_QLearning(env)
  fig, ax = plt.subplots()
  print(Q)
  print(Q.argmax(1))
  ax.plot(range(len(avg_rewards)), avg_rewards)
  fig.savefig("rewards.png")
  #render_single_Q(env, Q)

if __name__ == '__main__':
  pass
    #main()
