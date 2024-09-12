import logging
import random
import numpy as np
import pygame
import hydra
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb
from agents.agent import *
from utils.maker import make_env, make_agent, make_sensor_list
from utils.evaluation import Evaluator

# Global variables
envs = None


@hydra.main(version_base=None, config_path='./configs', config_name='inference')
def main(config: DictConfig) -> None:
    logging.info(f'Starting a new run @ {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.')

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Get some speed-up by using TensorCores on Nvidia Ampere GPUs
    # torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(config.setup.device_learning)

    # Update config
    OmegaConf.update(config, 'log_dir', HydraConfig.get().run.dir, force_add=True)
    OmegaConf.update(config, 'env.sensors.sensors', make_sensor_list(config.env.sensors.main_modality), force_add=True)

    logging.info(
        f'Using {device} as learning device.'
        f'Logdir is: {config.log_dir}.'
        f'Using {config.env.sensors.main_modality} as main modality.'
    )

    wandb_tag = f'{"gray_" if config.rl.image.grayscale else ""}' \
                f'{config.env.sensors.main_modality}_{"lstm" if config.rl.lstm.use else "frame"}'

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        sync_tensorboard=True,
        config=OmegaConf.to_container(config, resolve=True),
        dir=config.log_dir,
        mode='online' if config.wandb.use else 'disabled',
        save_code=True,
        tags=[wandb_tag]
    )
    # Make gym env(s)
    global envs
    envs = make_env(config)

    evaluator = Evaluator(config=config, envs=envs)

    # Make agent
    agent = make_agent(config, envs, device)
    data = torch.load(f'{config.setup.load_checkpoint.dir}/{config.setup.load_checkpoint.agent_name}.pt',
                      map_location='cuda:0')
    agent.load_state_dict(data['model_state_dict'], strict=True)
    evaluator.envs.set_obs_rms(data['obs_rms'])
    evaluator(agent, action_mode=config.eval.mode, n_steps=config.eval.n_steps, global_step=0, name='inference')
    envs.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.quit()
        if envs is not None:
            envs.close()
