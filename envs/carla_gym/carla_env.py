import copy
from collections import defaultdict
from skimage.transform import resize
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import cv2
from envs.carla_gym.misc import *
from envs.carla_gym.carla_manager import CarlaManager
from envs.carla_gym.vehicle_control.drive_dynamics_controller import *
from envs.carla_gym.vehicle_control.vehicle_control import VehicleController


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params, seed=None, alloc=None):
        self.alloc = alloc
        self.id = ''.join(str(x) for x in np.random.randint(0, 9, 5))
        self.n_vecs = None
        self.params = params
        self.seed = self.seed(seed)
        self.obs_size = params.env.sensors.size_output_image  # int(self.params.env.sensors.obs_range / self.params.env.sensors.lidar_bin)
        self.temp_ego_pos_anchor = None
        self.ego_pos = None

        self.reset_step = -1
        self.total_step = 0
        self.time_step = None
        self.total_resets = 0

        self.counter = 0

        # This starts the Carla server (sync mode) and connects the client
        self.carla_manager: CarlaManager = None
        self.ego_controller: VehicleController = None
        self.is_recording = False
        self.ego_traffic_light = None

        # Action space
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)

        # Observation space
        observation_space = dict()
        observation_space['state'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        n_dim_state = 6
        img_channel = 1 if params.rl.image.grayscale else 3

        for sensor in params.env.sensors.sensors:
            if sensor == 'RGBCamera':
                observation_space['camera'] = spaces.Box(
                    low=0, high=255, shape=(img_channel, self.obs_size, self.obs_size), dtype=np.uint8
                )
            elif sensor == 'Lidar':
                observation_space['lidar'] = spaces.Box(
                    low=0, high=255, shape=(img_channel, self.obs_size, self.obs_size), dtype=np.uint8
                )
            elif sensor == 'RGBBirdsEyeView':
                observation_space['rgb_birds_eye_view'] = spaces.Box(
                    low=0, high=255, shape=(img_channel, self.obs_size, self.obs_size), dtype=np.uint8
                )
            elif sensor == 'MultiBirdsEyeView':
                observation_space['multi_birds_eye_view'] = spaces.Box(
                    low=0, high=1, shape=(n_dim_state, self.obs_size, self.obs_size), dtype=np.float32
                )

            # Traffic and map features
            self.category2int = {'vehicle': 0, 'walker': 1}
            self.traffic_light_states = {carla.TrafficLightState.Red: 0,
                                         carla.TrafficLightState.Yellow: 1,
                                         carla.TrafficLightState.Green: 2,
                                         carla.TrafficLightState.Off: 3,
                                         carla.TrafficLightState.Unknown: 4}

            self.sign_cat2int = {'206': 0, '205': 1, '274': 2}  # stop: 206, yield: 205, speedlimit: 274
            # Traffic signs, see https://carla.readthedocs.io/en/latest/python_api/#carlalandmarktype
            self.carla_map = None

        self.observation_space = spaces.Dict(observation_space)
        self.display = None
        self.auto_done = None

    def start_experiment(self):
        if self.params.viz.pygame_rendering:
            self._init_renderer(self.params.viz.render_off_screen)
        self.carla_manager = CarlaManager(self.params, seed=self.seed, alloc=self.alloc)

    @property
    def world(self):
        return self.carla_manager.world

    @property
    def client(self):
        return self.carla_manager.client

    @property
    def ego_carla(self):
        return self.carla_manager.actor_manager.ego

    def _init_renderer(self, render_off_screen=False):
        """Initialize the birdeye view renderer."""
        if render_off_screen:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((1, 1))
        else:
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode(
                (self.params.viz.display_size * 3, self.params.viz.display_size),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

    def reset(self, seed=None, options=None):
        if options is not None and 'auto_done' in options:
            self.auto_done = options['auto_done']

        if self.carla_manager is None:
            # Start carla
            self.start_experiment()

        logging.debug("Env gym-carla will be reset.")
        self.total_resets += 1

        # Lightweight resetting: Reset only ego, other road users stay
        if self.ego_carla is not None:
            self.carla_manager.sensor_manager.close_all_sensors()
            self.carla_manager.actor_manager.clear_ego()

        logging.debug("Starting traffic: Init traffic manager and spawning other road users.")
        # Init tm only once (before other road users are spawned)
        if not self.carla_manager.tm_running:
            self.carla_manager.traffic_manager = self.carla_manager.init_traffic_manager()
            self.carla_manager.actor_manager.spawn_vehicles(self.params.env.sim.number_of_vehicles,
                                                            tm_port=self.carla_manager.tm_port)
            self.carla_manager.actor_manager.spawn_walkers(self.params.env.sim.number_of_walkers)

        self.carla_manager.actor_manager.spawn_ego(self.params)
        self.carla_manager.sensor_manager.spawn_ego_sensors(self.ego_carla)
        # self.carla_manager.traffic_manager.ignore_walkers_percentage(self.ego_carla, 0.0)

        # Respawn actors when they did get destroyed
        n_vehicles = len(self.world.get_actors().filter('vehicle.*'))

        if self.n_vecs is None:
            self.n_vecs = n_vehicles

        if self.n_vecs != n_vehicles:
            self.carla_manager.actor_manager.spawn_vehicles(self.n_vecs - n_vehicles,
                                                            tm_port=self.carla_manager.tm_port)
            n_vehicles = len(self.world.get_actors().filter('vehicle.*'))

        self.n_vecs = n_vehicles

        # Ego controller
        self.ego_controller = VehicleController(self.params, self.ego_carla)
        # Lateral control
        self.ego_controller.add_lateral_controller(StanleyLateralVehicleController)
        # self.ego_controller.add_lateral_controller(PIDLateralVehicleController)
        # self.lateral_controller.change_parameters(1.0, 0.1, 0.0, dt=0.1)
        # Longitudinal control
        # self.ego_controller.add_longitudinal_controller(PIDLongitudinalVehicleController)
        self.ego_controller.add_longitudinal_controller(PIDLongitudinalAccelerationVehicleController)
        # self.ego_controller.longitudinal_controller.change_parameters(0.3, 0.2, 0.1, dt=0.1)

        self.temp_ego_pos_anchor = np.array([self.ego_carla.get_location().x, self.ego_carla.get_location().y])
        self.ego_pos = None

        # Route planning: update waypoints
        self.ego_controller.update_state(self.world.get_snapshot())
        self.ego_controller.update_route()

        # Step the sim
        data = self.carla_manager.tick(waypoints=self.ego_controller.waypoints)

        self.ego_controller.update_state(self.world.get_snapshot())
        self.ego_controller.update_route()

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Update the view
        if not self.params.viz.render_off_screen and self.params.viz.follow_cam_ego:
            # Set spectator view to ego vehicle
            self.carla_manager.set_spectator_camera_view(
                self.ego_carla.get_transform(), z_offset=self.params.viz.spectator_z_offset)
        obs = self._get_obs(data)
        info = dict()
        return obs, info

    def step(self, action):
        """ Step the simulation forward and return the new observation, reward, done, info. Control action can be
        acceleration and steering, or only acceleration.
        Args:
        - action (np.ndarray): control action, e.g., [0.5, 0.0] for acceleration and steering, or [0.5] for only
        acceleration.
        Returns:
        - obs (np.ndarray): observation
        - reward (float): reward
        - done (bool): episode termination
        - info (dict): additional information
        """
        if action.size < 2:
            acceleration = action[0]
            steer = None
        else:
            acceleration = action[0]
            steer = action[1]

        ego_carla_action = self.ego_controller.get_carla_control_action(acceleration=acceleration, steer=steer)
        self.ego_controller.apply_carla_control_action(ego_carla_action)

        # Tick the simulation to move a step of length dt forward
        data = self.carla_manager.tick(waypoints=self.ego_controller.waypoints)

        if not self.params.viz.render_off_screen and self.params.viz.follow_cam_ego:
            self.carla_manager.set_spectator_camera_view(self.ego_carla.get_transform(),
                                                         z_offset=self.params.viz.spectator_z_offset)

        self.ego_controller.update_state(self.world.get_snapshot())
        self.ego_controller.update_route()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        # Return values
        obs = self._get_obs(data)
        reward, reward_comp = self._get_reward(data)
        terminated = self._terminated(data)
        truncated = self._truncated(data)

        if self.auto_done is not None and self.auto_done > 0:
            self.auto_done -= 1
            if self.auto_done == 0:
                terminated = True
                self.auto_done = None

        # Add info
        info = dict()
        info['ped_collision'] = copy.deepcopy(data['ped_collision'][1])
        info['car_collision'] = copy.deepcopy(data['car_collision'][1])
        info['collision'] = copy.deepcopy(data['collision'][1])
        info['run_red_light'] = copy.deepcopy(data['run_red_light'][1])
        info['ego_state'] = copy.deepcopy(data['ego_state'][1])
        info['distance_travelled'] = copy.deepcopy(data['distance_travelled'][1])

        # Add reward components to info
        for r_name, r_value in reward_comp.items():
            info[r_name] = copy.deepcopy(r_value)

        return obs, reward, terminated, truncated, copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def render(self, mode):
        pass

    def start_recorder(self, log_dir, name):
        # self.client.start_recorder(os.path.abspath(''.join(f"{log_dir}carla_{name}.log")), additional_data=False)
        # self.carla_manager.sensor_manager.start_recorder(log_dir, name)
        pass

    def stop_recorder(self, rec_dir):
        # self.client.stop_recorder()
        # data = self.carla_manager.sensor_manager.stop_recorder(rec_dir)
        # self.client.show_recorder_collisions("recording01.log", "h", "a")
        # self.client.show_recorder_file_info("recording01.log", False)
        # return data
        pass

    def close(self):
        if self.carla_manager is not None:
            self.carla_manager.close()
        super().close()
        pygame.display.quit()
        logging.debug('Gym carla env has been closed!')

    def _get_obs(self, data):
        obs = dict()
        # 0. Ego state
        obs['state'] = np.array(
            [
                data['dis_to_wps'][1],
                - data['yaw_to_wps'][1],
                data['ego_state'][1]['ego_vel_norm'],
                data['ego_state'][1]['ego_acc_norm'],
                data['ego_state'][1]['ego_steer'],
                data['ego_state'][1]['ego_speed_limit'],
                data['at_red_tfl'][1]
                # self.ego_controller.vehicle_front
            ],
            dtype=np.float32
        )

        # Sensor readouts
        # 1. Birdeye view image generation
        if 'RGBBirdsEyeView' in self.params.env.sensors.sensors:
            birdeye = data['rgb_birds_eye_view'][1]

            # Display birdeye image
            if self.params.viz.pygame_rendering:
                # birdeye_surface = birdeye[0:self.params.viz.display_size, 0:self.params.viz.display_size, :]
                birdeye_surface = rgb_to_display_surface(birdeye, self.params.env.sensors.size_output_image)
                birdeye_surface = pygame.transform.scale(birdeye_surface,
                                                         (self.params.viz.display_size, self.params.viz.display_size))
                self.display.blit(birdeye_surface, (0, 0))

            # make birdeye image grayscale using cv2
            if self.params.rl.image.grayscale:
                birdeye = cv2.cvtColor(birdeye, cv2.COLOR_RGB2GRAY)
                birdeye = np.expand_dims(birdeye, axis=-1)

            birdeye = birdeye.transpose((2, 0, 1))

            # import matplotlib.pyplot as plt
            # plt.imshow(birdeye.transpose(1,2,0)); plt.show()
            # plot if grayscale
            # plt.imshow(birdeye.squeeze(), cmap='gray'); plt.show()
            # bev = birdeye[0:self.obs_size, 0:self.obs_size, :]
            # bev = np.reshape(birdeye, (3, self.obs_size, self.obs_size))
            obs.update({'rgb_birds_eye_view': birdeye.astype(np.uint8)})

        if 'MultiBirdsEyeView' in self.params.env.sensors.sensors:
            multi_birdeye = data['multi_birds_eye_view'][1]
            # multi_birdeye_surface = multi_birdeye[0:self.params.viz.display_size, 0:self.params.viz.display_size, :]

            # Display birdeye image
            # if self.params.viz.pygame_rendering:
            #     birdeye_surface = rgb_to_display_surface(multi_birdeye, self.params.viz.display_size)
            #     self.display.blit(birdeye_surface, (0, 0))
            # Transform form (H, W, C) to (C, H, W)
            multi_birdeye = multi_birdeye.transpose((2, 0, 1))

            # bev = birdeye[0:self.obs_size, 0:self.obs_size, :]
            # bev = np.reshape(birdeye, (3, self.obs_size, self.obs_size))
            obs.update({'multi_birds_eye_view': multi_birdeye.astype(np.float32)})

        # 2. Lidar image generation
        if 'Lidar' in self.params.env.sensors.sensors:
            lidar = data['lidar'][1]
            # Display lidar image
            if self.params.viz.pygame_rendering:
                lidar_surface = rgb_to_display_surface(lidar * 255, self.params.viz.display_size)
                self.display.blit(lidar_surface, (self.params.viz.display_size, 0))

            if self.params.rl.image.grayscale:
                lidar = cv2.cvtColor(lidar, cv2.COLOR_RGB2GRAY)
                lidar = np.expand_dims(lidar, axis=-1)

            # Transform form (H, W, C) to (C, H, W)
            lidar = lidar.transpose((2, 0, 1))

            obs.update({'lidar': lidar.astype(np.uint8)})

        # 3. RGB Camera images
        if 'RGBCamera' in self.params.env.sensors.sensors:
            # Display camera image
            camera_img = data['rgb_camera'][1]  # self.carla_manager.sensor_manager.sensors['camera'].camera_img

            if camera_img.shape[0] != self.obs_size:
                camera = resize(camera_img, (self.obs_size, self.obs_size)) * 255
            else:
                camera = camera_img.astype(np.float32)

            if self.params.viz.pygame_rendering:
                camera_surface = rgb_to_display_surface(camera, self.params.viz.display_size)
                self.display.blit(camera_surface, (self.params.viz.display_size * 2, 0))

            # Transform form (H, W, C) to (C, H, W)
            if self.params.rl.image.grayscale:
                camera = cv2.cvtColor(camera, cv2.COLOR_RGB2GRAY)
                camera = np.expand_dims(camera, axis=-1)
            camera = camera.transpose((2, 0, 1))

            obs.update({'camera': camera.astype(np.uint8)})

        # Display on pygame
        if self.params.viz.pygame_rendering:
            pygame.display.flip()

        return obs

    def _get_actor_nodes(self, actor_ids):
        # Actor nodes
        actor_node_feats = np.zeros((self.max_nodes, self.actor_node_feat_dim), dtype=np.float32)
        actor_id2idx = defaultdict()

        for node_idx, actor_id in enumerate(actor_ids):
            # Construct and add node feature
            curr_actor_feat = get_actor_feat(self.world.get_actor(actor_id), self.ego_carla, self.category2int)
            actor_node_feats[node_idx] = curr_actor_feat
            actor_id2idx[actor_id] = node_idx

        return actor_node_feats, actor_id2idx

    def _get_reward(self, data):
        """Calculate the step reward."""

        # Collision
        r_collision = -1 if data['collision'][1] else 0

        # Longitudinal velocity
        vel_lon = data['ego_state'][1]['ego_vel_in_ego'][0]

        # Speeding
        r_speeding = -1 if data['ego_state'][1]['speeding'] else 0

        # Out of lane
        # GET distance from data
        r_out = -1 if abs(data['dis_to_wps'][1]) > self.params.rl.reward.out_lane_thres else 0
        # r_out = -abs(data['dis_to_wps'][1])

        # Steering:
        r_steer = - (data['ego_state'][1]['ego_steer'] ** 2)

        # Cost for lateral acceleration
        r_lat = - (abs(data['ego_state'][1]['ego_steer']) * vel_lon ** 2)

        # Speed tracking, not used for now
        r_speed_dev = - (abs(data['ego_state'][1]['speed_tracking_error']))

        # Running red light
        r_red_light = -1 if data['run_red_light'][1] else 0

        reward_coll = self.params.rl.reward.collision * r_collision
        reward_vel_long = self.params.rl.reward.vel_lon * vel_lon
        reward_speed = self.params.rl.reward.speeding * r_speeding
        reward_ool = self.params.rl.reward.oo_lane * r_out
        reward_steer = self.params.rl.reward.steer * r_steer
        reward_lat_acc = self.params.rl.reward.lat_acc * r_lat
        reward_speed_dev = self.params.rl.reward.speed_dev * r_speed_dev
        reward_red_light = self.params.rl.reward.red_light * r_red_light
        # Doesn't make sense since we're not having a goal state
        reward_r_step = - self.params.rl.reward.reward_r_step
        reward = \
            reward_coll + reward_speed_dev + reward_ool + reward_vel_long + reward_speed \
            + reward_steer + reward_lat_acc + reward_r_step

        if not self.params.rl.reward.no_traffic_lights:
            reward += reward_red_light

        reward_comp = dict({
            'reward_coll': reward_coll,
            'reward_vel_long': reward_vel_long,
            'reward_speed': reward_speed,
            'reward_speed_dev': reward_speed_dev,
            'reward_ool': reward_ool,
            'reward_steer': reward_steer,
            'reward_lat_acc': reward_lat_acc,
            'reward_red_light': reward_red_light * (not self.params.rl.reward.no_traffic_lights),
            'reward': reward,
        })
        if reward_steer > 10:
            print("\n\n ############## reward_steer is super large", reward_steer)

        # # Print rewards componentssna
        # for k, v in reward_comp.items():
        #     print(k, v)
        return reward, reward_comp

    def _terminated(self, data):
        """Terminates the episode if there was a collision."""
        if data['collision'][1]:
            return True
        elif not self.params.rl.reward.no_traffic_lights and data['run_red_light'][1]:
            return True
        return False

    def _truncated(self, data):
        """Truncates the episode if the maximum steps have been reached."""
        if self.time_step > self.params.rl.max_time_episode:
            return True
        return False

    def show_all_available_carla_cars(self):
        if self.world is not None:
            blueprint_library = self.world.get_blueprint_library()
            [print(bp.id) for bp in blueprint_library.filter('vehicle.*.*')]
        else:
            logging.error('Initialize a world first before all available carla cars can be printed.')
