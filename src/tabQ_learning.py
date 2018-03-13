import numpy as np
import gym
import time
import copy
from gym.envs import toy_text
from MultiEnv import make_multienv
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


  o_state = tuple(env.reset())
  e_orig = e
  q_alt = {o_state : np.zeros(env.nA)}
  avg_reward = 0
  rewards = np.zeros(num_episodes)
  avg_rewards = np.zeros(num_episodes)
  action_cts = [0,0]
  greedy_cts = [0,0]
  replay_buffer = []
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
        for z in range(env.n_envs):
          env.V[z] += (env.greedy_rollout(q_alt, z) - env.V[z])/float(env.n_iter[z] + 1)

      replay_buffer.append({'old_state' : old_state, 'state' : state, 'action' : action, 'done' : done, 'reward' : reward})
      old_state = state
      episode_reward += reward

      if len(replay_buffer) == 40:
        print("replay")
        for x in range(10):
          replay(q_alt, replay_buffer, 40, lr, gamma)
        replay_buffer = []
      print(q_alt[(0,0,0,0,0,0)])

    #q[old_state] = np.ones(env.nA)*reward
    #e = e_orig/(x)
  #lr = lr/float(x+1)
  return q_alt, avg_rewards, action_cts, greedy_cts, replay_buffer

def replay(q, replay_buffer, n, lr, gamma):
  transitions = np.random.choice(replay_buffer, size = n)
  for t in transitions:
    done, old_state, state, action, reward = t['done'], t['old_state'], t['state'], t['action'], t['reward']
    if not done:
      q[old_state][action] = q[old_state][action] + lr*(reward + gamma*q[state].max() - q[old_state][action])
    else:
      q[old_state][action] = q[old_state][action] + lr*(reward - q[old_state][action])


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
  mdps = [toy_text.NChainEnv(slip = x) for x in (0,1)]
  env = make_multienv(mdps, max_depth = 5, gamma = 1)
  learn_Q_QLearning(env)
  #render_single_Q(env, Q)

if __name__ == '__main__':
  pass
  #main()
