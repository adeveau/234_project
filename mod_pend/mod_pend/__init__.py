from gym.envs.registration import register

register(
    id='mod_pend-v0',
    entry_point='mod_pend.envs:ModPendulumEnv',
)
