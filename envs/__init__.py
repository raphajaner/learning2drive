from gymnasium.envs.registration import register

register(
    id='carla-v2',
    entry_point='envs.carla_gym:CarlaEnv',
)
