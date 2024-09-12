import numpy as np
import gymnasium as gym
import pynvml
import logging

from envs.carla_gym.carla_env import CarlaEnv
from envs.env_wrapper import NormalizeObservation
from envs.env_wrapper import NormalizeReward
from envs.carla_gym.misc import get_free_docker_ports
from agents.agent import *


def clip_over_list(data, clip_obs, exclude_keys=None):
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            subout = {}
            for subkey, subvalue in value.items():
                if subkey not in exclude_keys:
                    subout[subkey] = np.clip(
                        subvalue,
                        clip_obs[0],
                        clip_obs[1]
                    )
                else:
                    subout[subkey] = subvalue  # .astype(np.float32)
            out[key] = subout
        else:
            if key not in exclude_keys:
                out[key] = np.clip(
                    value,
                    clip_obs[0],
                    clip_obs[1]
                )
            else:
                out[key] = value  # .astype(np.float32)
    return out


def allocate(config):
    logging.info(f"Allocating CARLA instances to GPUs and free TCP ports")
    gpu_mem_size = None  # GPU memory size in MiB
    byte_to_mib = 1 / (1 / 1024 / 1024)
    carla_instance_mem = config.setup.carla_mem_instance_size  # MiB
    # used_mem_thresh = 0.1 * 10e9 / byte_to_mib # 10 MiB

    # Find all available GPUs with "free" memory
    available_gpus = list()
    all_device_idcs = [i for i in range(torch.cuda.device_count())]
    pynvml.nvmlInit()
    logging.info(f"Driver Version: {pynvml.nvmlSystemGetDriverVersion()}")
    for gpu_idx in all_device_idcs:
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_idx))
        # Get the general GPU memory size
        if gpu_mem_size is None:
            gpu_mem_size = gpu_mem.total / byte_to_mib
        # Check if sufficient GPU memory is available
        if gpu_mem.used < config.setup.gpu_used_thresh_factor * gpu_mem.total:
            available_gpus.append(gpu_idx)

    # Get intersection of generally available GPUs and selected host GPUs specified in config
    available_gpus = list(set(available_gpus).intersection(set(config.setup.host_gpus)))
    logging.info(f"Available GPUs: {available_gpus}")

    free_ports = get_free_docker_ports(base_port=config.setup.docker_base_port, num_ports=config.rl.num_envs * 3 * 2)
    logging.info(f"Free ports: {free_ports}")

    logging.info("Allocate envs to GPUs:")
    ports = list()
    max_num_envs_per_gpu = int(
        np.floor(gpu_mem_size / carla_instance_mem))  # we say a CARLA instance is approx. 6000 Mebibytes of memory
    gpu_env_counter = 0
    gpu_list_idx = 0
    port_idx = 0

    for env_idx in range(config.rl.num_envs):
        if gpu_env_counter == max_num_envs_per_gpu:
            gpu_env_counter = 0
            gpu_list_idx += 1

        if gpu_list_idx >= len(available_gpus):
            raise ValueError(
                "Not enough GPUs available for the number of environments. Try adjusting config.setup.gpu_used_thresh_factor ?")
        ports.append([free_ports[port_idx + 0],
                      free_ports[port_idx + 1],
                      free_ports[port_idx + 2],
                      available_gpus[gpu_list_idx]])
        port_idx += 6  # 3 (larger gap in between)
        gpu_env_counter += 1

    for env_ports in ports:
        logging.info(f"GPU ID: {env_ports[3]} - TCP: {env_ports[0:3]}")

    return ports


def _make_env(config, seed, gpu_alloc):
    def thunk():
        logging.info(f"Creating env with seed {seed}")
        env = CarlaEnv(params=config, seed=seed, alloc=gpu_alloc)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        from gymnasium.experimental.wrappers import FrameStackObservationV0
        if config.rl.frame_stack.use:
            env = FrameStackObservationV0(env, config.rl.frame_stack.n_frames)
        return env

    return thunk


def make_env(config):
    """Make the environment"""
    gym_vector_cls = gym.vector.AsyncVectorEnv if config.async_env else gym.vector.SyncVectorEnv

    # GPU Allocation
    if config.docker:
        gpu_alloc = allocate(config)
        envs = gym_vector_cls([_make_env(config, seed=i, gpu_alloc=gpu_alloc[i]) for i in range(config.rl.num_envs)])
    else:
        gpu_alloc = allocate(config)
        envs = gym_vector_cls([_make_env(config, seed=i, gpu_alloc=gpu_alloc[i]) for i in range(config.rl.num_envs)])

    # Observation wrapper
    if config.env.wrapper.normalize_obs:
        envs = NormalizeObservation(envs, exclude_keys=config.env.exclude_keys)
    if config.env.wrapper.clip_obs != "None":
        from functools import partial
        clip_over_list_partial = partial(clip_over_list,
                                         clip_obs=config.env.wrapper.clip_obs,
                                         exclude_keys=config.env.exclude_keys)
        envs = gym.wrappers.TransformObservation(envs, clip_over_list_partial)

    # Reward wrapper
    if config.env.wrapper.normalize_rew:
        envs = NormalizeReward(envs, gamma=config.rl.gamma)
    if config.env.wrapper.clip_rew != "None":
        envs = gym.wrappers.TransformReward(
            envs, lambda reward: np.clip(reward, config.env.wrapper.clip_rew[0], config.env.wrapper.clip_rew[1]))

    return envs


def make_sensor_list(main_modality):
    """Make the sensor list"""
    if main_modality == 'rgb_bev':
        return ['Collision', 'SemanticLidar', 'RGBBirdsEyeView', 'SemanticLidar360']
    elif main_modality == 'multi_bev':
        return ['Collision', 'SemanticLidar', 'MultiBirdsEyeView', 'SemanticLidar360']
    elif main_modality == 'rgb':
        return ['Collision', 'SemanticLidar', 'RGBCamera', 'SemanticLidar360']
    else:
        raise ValueError(f"Unknown main modality {main_modality}.")


def make_agent(config, envs, device):
    """Make the agent"""
    if config.env.sensors.main_modality == 'rgb_bev':
        if config.rl.lstm.use:
            if config.rl.image.grayscale:
                if config.rl.double_network:
                    agent = GrayBEVDoubleAgentLSTM(config, envs).to(device)
                else:
                    agent = GrayBEVAgentLSTM(config, envs).to(device)
            else:
                if config.rl.double_network:
                    agent = RGBBEVDoubleAgentLSTM(config, envs).to(device)
                else:
                    agent = RGBBEVAgentLSTM(config, envs).to(device)
        else:
            if config.rl.image.grayscale:
                if config.rl.double_network:
                    agent = GrayBEVDoubleAgent(config, envs).to(device)
                else:
                    agent = GrayBEVAgent(config, envs).to(device)
            else:
                if config.rl.double_network:
                    agent = RGBBEVDoubleAgent(config, envs).to(device)
                else:
                    agent = RGBBEVAgent(config, envs).to(device)
    elif config.env.sensors.main_modality == 'multi_bev':
        if config.rl.lstm.use:
            if config.rl.double_network:
                agent = MultiBEVDoubleAgentLSTM(config, envs).to(device)
            else:
                agent = MultiBEVAgentLSTM(config, envs).to(device)
        else:
            if config.rl.double_network:
                agent = MultiBEVDoubleAgent(config, envs).to(device)
            else:
                agent = MultiBEVAgent(config, envs).to(device)
    elif config.env.sensors.main_modality == 'rgb':
        if config.rl.lstm.use:
            if config.rl.double_network:
                agent = RGBDoubleAgentLSTM(config, envs).to(device)
            else:
                agent = RGBAgentLSTM(config, envs).to(device)
        else:
            if config.rl.double_network:
                agent = RGBDoubleAgent(config, envs).to(device)
            else:
                agent = RGBAgent(config, envs).to(device)
    else:
        raise NotImplementedError
    return agent
