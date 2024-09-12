from copy import deepcopy
from typing import Union, Dict

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.wrappers.NormalizeObservation):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, exclude_keys=['']):
        super().__init__(env, epsilon)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.is_dict_space = isinstance(self.single_observation_space, gym.spaces.dict.Dict)
        self.exclude_keys = exclude_keys
        self.obs_rms: Union[
            gym.wrappers.normalize.RunningMeanStd, Dict[str, gym.wrappers.normalize.RunningMeanStd], None] = None
        if self.is_vector_env:
            if self.is_dict_space:
                self.obs_rms = {}
                for key, space in self.single_observation_space.items():
                    if isinstance(space, gym.spaces.dict.Dict):
                        rms = {}
                        for subkey, subspace in space.items():
                            if subkey not in exclude_keys:
                                rms[subkey] = gym.wrappers.normalize.RunningMeanStd(shape=subspace.shape)
                        self.obs_rms[key] = rms
                    else:
                        if key not in exclude_keys:
                            self.obs_rms[key] = gym.wrappers.normalize.RunningMeanStd(shape=space.shape)

            else:
                self.obs_rms = gym.wrappers.normalize.RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = gym.wrappers.normalize.RunningMeanStd(shape=self.observation_space.shape)
        self._block_update_obs = False

    @property
    def block_update_obs(self):
        return self._block_update_obs

    def set_block_update_obs(self, value):
        """Blocks the update of the running mean and variance of the observations.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_obs = value

    def get_obs_rms(self):
        return deepcopy(self.obs_rms)

    def set_obs_rms(self, value):
        """Sets the running mean and variance of the observations."""
        self.obs_rms = value

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        if self.is_dict_space:
            out = {}
            for key, value in obs.items():
                if isinstance(value, dict):
                    subout = {}
                    sub_rms = self.obs_rms[key]
                    for subkey, subvalue in value.items():
                        if subkey not in self.exclude_keys:
                            if not self._block_update_obs:
                                sub_rms[subkey].update(subvalue)
                            subout[subkey] = ((subvalue - sub_rms[subkey].mean) / np.sqrt(
                                sub_rms[subkey].var + self.epsilon)).astype(np.float32)
                        else:
                            subout[subkey] = subvalue.astype(np.float32)
                    out[key] = subout
                else:
                    if key not in self.exclude_keys:
                        if not self._block_update_obs:
                            self.obs_rms[key].update(value)
                        out[key] = ((value - self.obs_rms[key].mean) / np.sqrt(
                            self.obs_rms[key].var + self.epsilon)).astype(np.float32)
                    else:
                        out[key] = value.astype(np.float32)
        else:
            if not self._block_update_obs:
                self.obs_rms.update(obs)
            out = ((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)).astype(np.float32)

        return out


class NormalizeReward(gym.wrappers.NormalizeReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_update_rew = False
        self._block_calc_rew = False

    @property
    def block_update_rew(self):
        """Variable to block the update of the running mean and variance of the rewards."""
        return self._block_update_rew

    def set_block_update_rew(self, value):
        """Blocks the update of the running mean and variance of the rewards.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_rew = value

    @property
    def block_calc_rew(self):
        return self._block_calc_rew

    def set_block_calc_rew(self, value):
        self._block_calc_rew = value

    def get_rew_rms(self):
        """Returns the running mean and variance of the rewards."""
        return deepcopy(self.return_rms)

    def set_rew_rms(self, value):
        """Sets the running mean and variance of the rewards."""
        self.return_rms = value

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if not self._block_update_rew:
            self.return_rms.update(self.returns)
        if self._block_calc_rew:
            # print(f"rews are blocked: {rews}.")
            return rews
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        # self.returns *= (1 - np.asarray(terminateds, dtype=np.float32))
        self.returns *= (1 - terminateds)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos
