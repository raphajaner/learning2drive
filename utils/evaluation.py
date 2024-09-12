import numpy as np
from tensordict import TensorDict
import torch
import wandb
from copy import deepcopy


class RolloutRecording:
    def __init__(self):
        # List tracing the history of full episodes
        self.h = []
        # Dict tracing the history of the current episode
        self.current_h = self._init_dict()

    @staticmethod
    def _init_dict():
        return dict(
            reward=[],
            reward_coll=[],
            reward_vel_long=[],
            reward_speed=[],
            reward_speed_dev=[],
            reward_ool=[],
            reward_steer=[],
            reward_lat_acc=[],
            reward_red_light=[],
            collision=[],
            ped_collision=[],
            car_collision=[],
            run_red_light=[],
            ego_state=[],
            distance_travelled=[],
        )

    def append(self, new_episode: bool, reward,
               reward_coll, reward_vel_long, reward_speed, reward_speed_dev, reward_ool, reward_steer, reward_lat_acc,
               reward_red_light, collision, ped_collision, car_collision, run_red_light, ego_state, distance_travelled):
        if reward is not None:
            self.current_h['reward'].append(reward)
            self.current_h['reward_coll'].append(reward_coll)
            self.current_h['reward_vel_long'].append(reward_vel_long)
            self.current_h['reward_speed'].append(reward_speed)
            self.current_h['reward_speed_dev'].append(reward_speed_dev)
            self.current_h['reward_ool'].append(reward_ool)
            self.current_h['reward_steer'].append(reward_steer)
            self.current_h['reward_lat_acc'].append(reward_lat_acc)

        self.current_h['collision'].append(collision)
        self.current_h['ped_collision'].append(ped_collision)
        self.current_h['car_collision'].append(car_collision)
        self.current_h['run_red_light'].append(run_red_light)
        self.current_h['ego_state'].append(ego_state)
        self.current_h['distance_travelled'].append(distance_travelled)

        # The next set added will be the start of a new eps
        if new_episode:
            self.h.append(deepcopy(self.current_h))
            self.current_h = self._init_dict()


class Evaluator:
    def __init__(self, config, envs, writer=None) -> None:
        self.config = config
        self.envs = envs.env

    @torch.inference_mode()
    def __call__(self, agent, n_steps, action_mode, global_step=0, name='eval'):
        # Inference mode
        agent.train(False)
        self.envs.set_block_update_rew(True)
        self.envs.set_block_update_obs(True)
        self.envs.set_block_calc_rew(True)

        print('Evaluating agent...')
        history = vec_rollout(self.config, self.envs, agent, n_steps=n_steps, action_mode=action_mode)
        print('Done evaluating agent.')

        agent.train(True)
        self.envs.set_block_update_rew(False)
        self.envs.set_block_update_obs(False)
        self.envs.set_block_calc_rew(False)

        # Evaluation metrics
        reward = np.mean([np.mean(h_['reward']) for h in history for h_ in h.h])
        reward_coll = np.mean([np.mean(h_['reward_coll']) for h in history for h_ in h.h])
        reward_vel_long = np.mean([np.mean(h_['reward_vel_long']) for h in history for h_ in h.h])
        reward_speed = np.mean([np.mean(h_['reward_speed']) for h in history for h_ in h.h])
        reward_speed_dev = np.mean([np.mean(h_['reward_speed_dev']) for h in history for h_ in h.h])
        reward_ool = np.mean([np.mean(h_['reward_ool']) for h in history for h_ in h.h])
        reward_steer = np.mean([np.mean(h_['reward_steer']) for h in history for h_ in h.h])
        reward_lat_acc = np.mean([np.mean(h_['reward_lat_acc']) for h in history for h_ in h.h])
        reward_red_light = np.mean([np.mean(h_['reward_red_light']) for h in history for h_ in h.h])

        collision = np.sum([np.sum(h_['collision']) for h in history for h_ in h.h])
        ped_collision = np.sum([np.sum(h_['ped_collision']) for h in history for h_ in h.h])
        car_collision = np.sum([np.sum(h_['car_collision']) for h in history for h_ in h.h])
        run_red_light = np.sum([np.sum(h_['run_red_light']) for h in history for h_ in h.h])
        distance_travelled = np.sum([np.sum(h_['distance_travelled']) for h in history for h_ in h.h])

        collisions_over_distance = collision / distance_travelled
        ped_collisions_over_distance = ped_collision / distance_travelled
        car_collisions_over_distance = car_collision / distance_travelled
        red_lights_over_distance = run_red_light / distance_travelled
        infractions_over_distance = collisions_over_distance + red_lights_over_distance

        ego_vel = np.mean([s['ego_vel_norm'] for h in history for h_ in h.h for s in h_['ego_state']])
        # take only the ego_vel that are larger than 0.2 m/s
        ego_vel_drive = np.mean(
            [s['ego_vel_norm'] for h in history for h_ in h.h for s in h_['ego_state'] if s['ego_vel_norm'] > 0.2])
        speed_tracking_error = np.mean(
            [np.abs(s['speed_tracking_error']) for h in history for h_ in h.h for s in h_['ego_state']])
        speeding = np.mean([s['speeding'] for h in history for h_ in h.h for s in h_['ego_state']])

        print(
            f'Evaluation:\n'
            f'- reward={reward:0.4f}\n'
            f'- reward_coll={reward_coll:0.4f}\n'
            f'- collision={collision}\n'
            f'- ped_collision={ped_collision}\n'
            f'- car_collision={car_collision}\n'
            f'- run_red_light={run_red_light}\n'
            f'- distance_travelled={distance_travelled:0.2f}\n'
            f'- ego_vel={ego_vel:0.2f}\n'
            f'- ego_vel_drive02={ego_vel_drive:0.2f}\n'
            f'- speed_tracking_error={speed_tracking_error:0.2f}\n'
            f'- speeding={speeding:0.6f}\n'
            f'- collisions_over_distance={collisions_over_distance:0.6f}\n'
            f'- ped_collisions_over_distance={ped_collisions_over_distance:0.6f}\n'
            f'- car_collisions_over_distance={car_collisions_over_distance:0.6f}\n'
            f'- red_lights_over_distance={red_lights_over_distance:0.6f}\n'
            f'- infractions_over_distance={infractions_over_distance:0.6f}\n'
        )

        if action_mode:
            name = 'mode_' + name

        wandb.log({
            f"{name}/global_step": global_step,
            f"{name}/reward": reward,
            f"{name}/reward_coll": reward_coll,
            f"{name}/reward_vel_long": reward_vel_long,
            f"{name}/reward_speed": reward_speed,
            f"{name}/reward_speed_dev": reward_speed_dev,
            f"{name}/reward_ool": reward_ool,
            f"{name}/reward_steer": reward_steer,
            f"{name}/reward_lat_acc": reward_lat_acc,
            f"{name}/reward_red_light": reward_red_light,
            f"{name}/collision": collision,
            f"{name}/ped_collision": ped_collision,
            f"{name}/car_collision": car_collision,
            f"{name}/run_red_light": run_red_light,
            f"{name}/distance_travelled": distance_travelled,
            f"{name}/ego_vel": ego_vel,
            f"{name}/ego_vel_drive02": ego_vel_drive,
            f"{name}/speed_tracking_error": speed_tracking_error,
            f"{name}/speeding": speeding,
            f"{name}/collisions_over_distance": collisions_over_distance,
            f"{name}/ped_collisions_over_distance": ped_collisions_over_distance,
            f"{name}/car_collisions_over_distance": car_collisions_over_distance,
            f"{name}/red_lights_over_distance": red_lights_over_distance,
            f"{name}/infractions_over_distance": infractions_over_distance,
        })
        return infractions_over_distance, ego_vel


@torch.inference_mode()
def vec_rollout(config, envs, agent, n_steps, action_mode=True):
    """ Rollout the agent in the environment for `config.eval.n_eps` episodes
    Args:
        config: config object
        envs: vectorized environment (reset automatically)
        agent: agent object
        action_mode: if True, action is the distribution's mode, else it is sampled
    Returns:
        history: list of RolloutRecording objects
    """
    history = [RolloutRecording() for _ in range(envs.num_envs)]
    device = agent.device

    next_obs, infos = envs.reset(options={'auto_done': n_steps})  # options={'rec_dir': log_dir})
    next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    next_lstm_state = agent.get_initial_state(envs.num_envs)

    for i in range(n_steps):
        next_obs = TensorDict(next_obs, [next_obs['state'].shape[0]], device=device)
        if config.rl.frame_stack.use:
            next_obs = next_obs.apply(lambda tensor: torch.cat(
                [tensor[:, i, ...] for i in range(0, tensor.shape[1], config.rl.frame_stack.skip_frames)], 1))
        action, next_lstm_state = agent.get_action(next_obs, next_lstm_state, next_done, mode=action_mode)
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_terminated = torch.tensor(terminated, device=device)
        next_truncated = torch.tensor(truncated, device=device)
        next_done = torch.logical_or(next_terminated, next_truncated)

        if 'final_info' in infos:
            for i in range(envs.num_envs):
                if infos['_final_info'][i]:
                    history[i].append(
                        True,
                        reward[i],
                        infos['final_info'][i]['reward_coll'],
                        infos['final_info'][i]['reward_vel_long'],
                        infos['final_info'][i]['reward_speed'],
                        infos['final_info'][i]['reward_speed_dev'],
                        infos['final_info'][i]['reward_ool'],
                        infos['final_info'][i]['reward_steer'],
                        infos['final_info'][i]['reward_lat_acc'],
                        infos['final_info'][i]['reward_red_light'],
                        infos['final_info'][i]['collision'],
                        infos['final_info'][i]['ped_collision'],
                        infos['final_info'][i]['car_collision'],
                        infos['final_info'][i]['run_red_light'],
                        infos['final_info'][i]['ego_state'],
                        infos['final_info'][i]['distance_travelled'],
                    )
                    # Don't add the first obs (initial obs) to the history
                else:
                    history[i].append(
                        False,
                        reward[i],
                        infos['reward_coll'][i],
                        infos['reward_vel_long'][i],
                        infos['reward_speed'][i],
                        infos['reward_speed_dev'][i],
                        infos['reward_ool'][i],
                        infos['reward_steer'][i],
                        infos['reward_lat_acc'][i],
                        infos['reward_red_light'][i],
                        infos['collision'][i],
                        infos['ped_collision'][i],
                        infos['car_collision'][i],
                        infos['run_red_light'][i],
                        infos['ego_state'][i],
                        infos['distance_travelled'][i],
                    )
        else:
            for i in range(envs.num_envs):
                history[i].append(
                    False,
                    reward[i],
                    infos['reward_coll'][i],
                    infos['reward_vel_long'][i],
                    infos['reward_speed'][i],
                    infos['reward_speed_dev'][i],
                    infos['reward_ool'][i],
                    infos['reward_steer'][i],
                    infos['reward_lat_acc'][i],
                    infos['reward_red_light'][i],
                    infos['collision'][i],
                    infos['ped_collision'][i],
                    infos['car_collision'][i],
                    infos['run_red_light'][i],
                    infos['ego_state'][i],
                    infos['distance_travelled'][i],
                )

    return history
