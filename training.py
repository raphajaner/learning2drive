import copy
import random
import time
import pygame
import torch.optim as optim
from tensordict import TensorDict
import hydra
from tqdm import tqdm

from agents.loss import PPOLoss
from agents.utils import save_model_summary
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from utils.maker import *
from utils.evaluation import Evaluator
import wandb

# Global variables
envs = None


@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(config: DictConfig) -> None:
    logging.info(f'Starting a new run @ {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.')

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Get some speed-up by using TensorCores on Nvidia Ampere GPUs
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(config.setup.device_learning)

    # Update config
    OmegaConf.update(config, 'log_dir', HydraConfig.get().run.dir, force_add=True)
    OmegaConf.update(config, 'batch_size', int(config.rl.num_envs * config.rl.num_steps), force_add=True)
    OmegaConf.update(config, 'minibatch_size', int(config.batch_size // config.rl.num_minibatches), force_add=True)
    OmegaConf.update(config, 'env.sensors.sensors', make_sensor_list(config.env.sensors.main_modality), force_add=True)
    OmegaConf.update(config, 'num_updates', int(config.rl.total_timesteps // config.batch_size), force_add=True)

    logging.info(
        f'Using {device} as learning device.'
        f'Logdir is: {config.log_dir}.'
        f'Using {config.env.sensors.main_modality} as main modality.'
        f'Full batch size: {config.batch_size}; '
        f'mini batch size {config.minibatch_size}; '
        f'update epochs: {config.rl.update_epochs} -> '
        f'num_updates: {config.num_updates}.'
    )

    wandb_tag = f'{"gray_" if config.rl.image.grayscale else ""}' \
                f'{config.env.sensors.main_modality}_{"lstm" if config.rl.lstm.use else "frame"}'

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
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

    # Evaluator for testing
    evaluator = Evaluator(config=config, envs=envs)
    best_infractions_over_distance = np.inf
    best_ego_vel = 0
    best_global_step = 0
    best_weights = None
    best_obs_rms = None

    # Make agent
    agent = make_agent(config, envs, device)
    if config.rl.lstm.use:
        next_lstm_state = agent.get_initial_state(config.rl.num_envs)
        next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    else:
        next_lstm_state, next_done = (False, False)

    ppo_loss = PPOLoss(config).to(device)

    if int(torch.__version__[0]) >= 2 and config.compile_torch:
        if config.env.sensors.main_modality == 'bev' or config.env.sensors.main_modality == 'rgb':
            agent = torch.compile(agent)
        elif config.env.sensors.main_modality == 'graph':
            import torch_geometric
            agent = torch_geometric.compile(agent)
        ppo_loss = torch.compile(ppo_loss)

    # Init lazy layers
    with torch.no_grad():
        next_obs, _ = envs.reset(seed=config.seed)
        next_obs = TensorDict(next_obs, [next_obs['state'].shape[0]], device=device)
        if config.rl.frame_stack.use:
            next_obs = next_obs.apply(lambda tensor: torch.cat(
                [tensor[:, i, ...] for i in range(0, tensor.shape[1], config.rl.frame_stack.skip_frames)], 1))
        agent.get_action_and_value(next_obs, None, next_lstm_state, next_done)

    save_model_summary(log_dir=wandb.run.dir, models=[agent])
    wandb.save("model_summary.txt")

    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.rl.learning_rate,
        betas=(0.9, 0.999),  # default values
        eps=1e-5,  # default value, ppo paper uses 1e-5
        weight_decay=config.rl.weight_decay  # default value
        # fused=True  # potential speedup but new
    )
    if config.setup.load_checkpoint.use:
        data = torch.load(f'{config.setup.load_checkpoint.dir}/agent.pt', map_location='cuda:0')
        agent.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        evaluator.envs.set_obs_rms(data['obs_rms'])
        epoch_start = data['epoch']
    else:
        epoch_start = 1

    # ALGO Logic: Storage setup
    obs = TensorDict({}, [config.rl.num_steps, config.rl.num_envs], device=device)
    actions = torch.zeros((config.rl.num_steps, config.rl.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device)
    rewards = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device)
    values = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device)
    terminateds = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device, dtype=torch.bool)
    truncateds = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device, dtype=torch.bool)
    dones = torch.zeros((config.rl.num_steps, config.rl.num_envs), device=device, dtype=torch.bool)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = start_time_rel = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = TensorDict(next_obs, [next_obs['state'].shape[0]], device=device)
    if config.rl.frame_stack.use:
        next_obs = next_obs.apply(lambda tensor: torch.cat(
            [tensor[:, i, ...] for i in range(0, tensor.shape[1], config.rl.frame_stack.skip_frames)], 1))
    next_terminated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    next_truncated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)

    gamma = torch.tensor(config.rl.gamma, device=device, requires_grad=False)
    gae_lambda = torch.tensor(config.rl.gae_lambda, device=device, requires_grad=False)
    # clip_coef = torch.tensor(config.rl.clip_coef, device=device, requires_grad=False)
    # ent_coef = torch.tensor(config.rl.ent_coef, device=device, requires_grad=False)
    # vf_coef = torch.tensor(config.rl.vf_coef, device=device, requires_grad=False)

    # Scheduler for lr
    if config.rl.anneal_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_updates + 1,
            eta_min=config.rl.anneal_lr_factor * optimizer.param_groups[0]["lr"]
        )

    for update in tqdm(range(epoch_start, config.num_updates + 1), desc='Update steps', colour='yellow'):

        if config.env.wrapper.block_updates:
            if update > config.env.wrapper.block_after_n_updates:
                envs.set_block_update_rew(True)
                envs.set_block_update_obs(True)
                assert envs.block_update_rew and envs.block_update_obs, \
                    "Updating normalization stats couldn't be blocked."
            else:
                assert not envs.block_update_rew and not envs.block_update_obs, \
                    "Updating normalization stats couldn't be blocked."

        if config.rl.lstm.use:
            if config.rl.double_network:
                initial_lstm_state = (deepcopy(next_lstm_state[0]), deepcopy(next_lstm_state[1]))
            else:
                initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # Generate trajectories by interaction with the environment
        for step in tqdm(range(0, config.rl.num_steps), desc='Data collection', colour='blue'):
            global_step += 1 * config.rl.num_envs
            obs[step] = next_obs
            truncateds[step] = next_truncated
            terminateds[step] = next_terminated
            dones[step] = next_done

            if torch.isnan(rewards).any() or any(
                    [torch.isnan(d).any() for d in next_obs.flatten_keys().to_dict().values()]):
                # NaN came from inside the environment (not from normalization) because the reward calculation
                # sometimes leads to NaNs when there was an error with the out-ouf-lane reward calculation
                rewards = torch.where(torch.isnan(rewards), torch.zeros_like(rewards), rewards)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_offset = 0.2 if step < config.rl.actor.n_steps_action_offset else None
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, None, next_lstm_state, next_done, action_offset=action_offset)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_terminated = torch.tensor(terminated, device=device)
            next_truncated = torch.tensor(truncated, device=device)
            next_done = torch.logical_or(next_terminated, next_truncated)
            # Only the obs are saved in a TensorDict
            next_obs = TensorDict(next_obs, [next_obs['state'].shape[0]], device=device)

            if config.rl.frame_stack.use:
                next_obs = next_obs.apply(lambda tensor: torch.cat(
                    [tensor[:, i, ...] for i in range(0, tensor.shape[1], config.rl.frame_stack.skip_frames)], 1))
                # import matplotlib.pyplot as plt
                # import pdb;
                # pdb.set_trace()
                # plt.imshow(next_obs['rgb_birds_eye_view'][0, 0, ...].cpu().numpy(), cmap='gray')
                # plt.show()

            if "final_info" in infos.keys():
                episodic_return = []
                episodic_length = []
                for item, final_info in zip(infos["final_info"], infos["_final_info"]):
                    if final_info:
                        episodic_return.append(item["episode"]["r"])
                        episodic_length.append(item["episode"]["l"])
                wandb.log({
                    "episodic/return": np.mean(episodic_return),
                    "episodic/length": np.mean(episodic_length),
                    "episodic/global_step": global_step
                })
        if update < config.env.wrapper.n_warmup:
            continue

        # bootstrap value if not done
        with torch.no_grad():
            agent.train(False)
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(config.rl.num_steps)):
                if t == config.rl.num_steps - 1:
                    # nextnonterminal = 1 - (next_done.int() - next_truncated.int())
                    nextnonterminal = ~next_terminated
                    nextnondone = ~next_done
                    nextvalues = next_value

                else:
                    # nextnonterminal = 1 - (dones[t + 1].int() - truncateds[t + 1].int())
                    nextnonterminal = ~terminateds[t + 1]
                    nextnondone = ~dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal.float() - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnondone.float() * lastgaelam
            returns = advantages + values
            agent.train(True)

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # Flatten over sequence X env dimension: (seq_len, envs, obs_dim) -> (seq_len*envs, obs_dim)
        # Order is: [env1[0], env2[0], env3[0], ..., env1[1], env2[1], env3[1], ...]
        b_obs = obs.view(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        clipfracs = []

        # Optimizing the policy and value network
        if config.rl.lstm.use:
            assert config.rl.num_envs % config.rl.num_minibatches == 0, \
                "Number of envs must be divisible by number of minibatches"
            envsperbatch = config.rl.num_envs // config.rl.num_minibatches
            b_inds = np.arange(config.rl.num_envs)  # Index of the envs (=batch)
            flatinds = np.arange(config.batch_size).reshape(config.rl.num_steps, config.rl.num_envs)
            batch_size = config.rl.num_envs  # Because we split/minibatch over envs
            minibatch_size = envsperbatch  # How many envs per minibatch to use
            num_sub_minibatches = config.rl.lstm.num_sub_minibatches  # How many sub minibatches to use for TBPTT
            assert config.rl.num_steps % num_sub_minibatches == 0, \
                "Number of steps must be divisible by number of sub minibatches"
            sub_minibatch_size = config.rl.num_steps // num_sub_minibatches  # How many steps per sub minibatch
        else:
            b_inds = np.arange(config.batch_size)
            batch_size = config.batch_size
            minibatch_size = config.minibatch_size

        for epoch in tqdm(range(config.rl.update_epochs), desc='Training epochs', colour='green'):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size

                if config.rl.lstm.use:
                    mbenvinds = b_inds[start:end]

                    if config.rl.double_network:
                        # detach gradients of hidden state which is a dict
                        hidden_state = (
                            dict(
                                actor=initial_lstm_state[0]['actor'][:, mbenvinds],
                                critic=initial_lstm_state[0]['critic'][:, mbenvinds]
                            ),
                            dict(
                                actor=initial_lstm_state[1]['actor'][:, mbenvinds],
                                critic=initial_lstm_state[1]['critic'][:, mbenvinds]
                            )
                        )
                    else:
                        hidden_state = (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds])

                    for sub_start in range(0, config.rl.num_steps, sub_minibatch_size):
                        sub_end = sub_start + sub_minibatch_size
                        mb_inds = flatinds[sub_start:sub_end, mbenvinds].ravel()  # be really careful about the index
                        _, newlogprob, entropy, newvalue, hidden_state = agent.get_action_and_value(
                            x=b_obs[mb_inds],
                            action=b_actions[mb_inds],
                            hidden_state=hidden_state,
                            done=b_dones[mb_inds]
                        )

                        loss, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfrac, grads = \
                            ppo_loss.grad_step(
                                config, agent, optimizer,
                                b_returns[mb_inds], b_values[mb_inds], b_advantages[mb_inds], b_logprobs[mb_inds],
                                newlogprob, entropy, newvalue
                            )

                        # Detach hidden state from graph for correct truncated backpropagation through time (TBPTT)
                        if config.rl.double_network:
                            # detach gradients of hidden state which is a dict
                            hidden_state = (
                                dict(
                                    actor=hidden_state[0]['actor'].detach(), critic=hidden_state[0]['critic'].detach()
                                ),
                                dict(
                                    actor=hidden_state[1]['actor'].detach(), critic=hidden_state[1]['critic'].detach()
                                )
                            )
                        else:
                            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                        clipfracs += [clipfrac.float().mean().item()]

                        if config.rl.target_kl != 'None':
                            if approx_kl > config.rl.target_kl:
                                break

                else:
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        x=b_obs[mb_inds],
                        action=b_actions[mb_inds],
                        hidden_state=None,
                        done=None
                    )

                    loss, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfrac, grads = \
                        ppo_loss.grad_step(
                            config, agent, optimizer,
                            b_returns[mb_inds], b_values[mb_inds], b_advantages[mb_inds], b_logprobs[mb_inds],
                            newlogprob, entropy, newvalue
                        )

                    clipfracs += [clipfrac.float().mean().item()]

            if config.rl.target_kl != 'None':
                if approx_kl > config.rl.target_kl:
                    break

        # Update lr_scheduler
        if config.rl.anneal_lr:
            lr_scheduler.step()

        # decrease agent.std_dev linearly to 0.1 over the course of training starting from the initial value 0.5
        if config.rl.anneal_actor_logstd:
            agent.actor_logstd.data = torch.tensor(
                max(-2, agent.actor_logstd - (config.rl.init_logstd + 2) / config.num_updates), device=device,
                requires_grad=False)

        # Metrics and logging of the training
        current_time = time.time()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        rmse_y = np.sqrt(np.mean((y_true - y_pred) ** 2))

        wandb.log({
            "charts/global_step": global_step,
            "charts/SPS_global": int(global_step / (current_time - start_time)),
            "charts/SPS": int(global_step / (current_time - start_time_rel)),
            "charts/UPS_global": int(update / (current_time - start_time_rel)),
            "charts/UPS": int(update / (current_time - start_time_rel)),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "losses/rmse_y": rmse_y,
            "losses/epochs": epoch,
            "losses/grads": grads,
            "losses/learning_rate": optimizer.param_groups[0]["lr"],
            "batch/rewards_mean": rewards.detach().mean().item(),
            "batch/rewards_std": rewards.detach().std().item(),
            "batch/rewards_min": rewards.detach().min().item(),
            "others/actor_logstd": torch.exp(agent.get_parameter('actor_logstd').detach()).item(),
            "others/return_rms_mean": envs.get_rew_rms().mean,
            "others/return_rms_std": envs.get_rew_rms().var,
            "others/return_rms_count": envs.get_rew_rms().count,
        })

        start_time_rel = current_time
        if update % config.eval.freq == 0 and update > 0 or update == config.num_updates:
            infractions_over_distance, ego_vel = evaluator(agent, action_mode=config.eval.mode,
                                                           n_steps=config.eval.n_steps,
                                                           global_step=global_step)
            if ego_vel > 1.0 and (
                    (infractions_over_distance < best_infractions_over_distance)
                    or (np.abs(infractions_over_distance - best_infractions_over_distance) / (
                    best_infractions_over_distance + 1e6) < 0.1
                        and ego_vel > best_ego_vel)):
                best_infractions_over_distance = infractions_over_distance
                best_ego_vel = ego_vel
                best_global_step = global_step
                best_weights = copy.deepcopy(agent.state_dict())
                best_obs_rms = copy.deepcopy(envs.get_obs_rms())
                wandb.run.summary["best_infractions_over_distance"] = best_infractions_over_distance
                wandb.run.summary["best_ego_vel"] = best_ego_vel
                wandb.run.summary["best_global_step"] = best_global_step
                logging.info(
                    f"New best model at global step {best_global_step} with {best_infractions_over_distance} "
                    f"infractions over distance and {best_ego_vel} ego velocity."
                )

            # Reset everything after evaluation to ensure correct states
            next_obs, info = envs.reset(seed=config.seed)
            next_obs = TensorDict(next_obs, [next_obs['state'].shape[0]], device=device)
            if config.rl.frame_stack.use:
                next_obs = next_obs.apply(lambda tensor: torch.cat(
                    [tensor[:, i, ...] for i in range(0, tensor.shape[1], config.rl.frame_stack.skip_frames)], 1))
            next_lstm_state = agent.get_initial_state(config.rl.num_envs)
            next_terminated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
            next_truncated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
            next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)

            # # Save current agent to disk
            # torch.save(
            #     {
            #         'update': update,
            #         'model_state_dict': agent.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            #         'loss': loss,
            #         'obs_rms': envs.get_obs_rms()
            #     },
            #     f'{wandb.run.dir}/agent_{global_step}.pt'
            # )
            # # Sync torch models immediately when written to wandb.run.dir
            # wandb.save("*.pt")

    # Evaluation of the best agent (saved to disk)
    agent.load_state_dict(best_weights)
    envs.set_obs_rms(best_obs_rms)
    torch.save(
        {
            'model_state_dict': agent.state_dict(),
            'obs_rms': envs.get_obs_rms()
        },
        f'{wandb.run.dir}/agent.pt'
    )
    wandb.save("*.pt")
    evaluator(agent, n_steps=80000, action_mode=config.eval.mode, global_step=global_step, name='best_model')


if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.quit()
        if envs is not None:
            envs.close()
