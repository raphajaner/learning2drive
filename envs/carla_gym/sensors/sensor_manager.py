from envs.carla_gym.sensors.sensors import *
from envs.carla_gym.collision_models import collision_hero_to_multiple


class SensorManager:
    """ Class to manage sensors in the environment. """

    def __init__(self, params, client):
        self.params = params
        world = client.get_world()
        self.world = world
        self._sensors = dict()
        self._sensor_queues = dict()
        self._event_sensor_queues = dict()
        self.ego = None
        self.ego_traffic_light = None

        # Initialize the blueprints
        self.sensor_bps = dict()
        for sensor_type in self.params.env.sensors.sensors:
            if sensor_type == 'RGBCamera':
                self.sensor_bps['camera'] = RGBCamera.create_bp(self.params, world)
            elif sensor_type == 'Lidar':
                self.sensor_bps['lidar'] = Lidar.create_bp(self.params, world)
            elif sensor_type == 'Collision':
                self.sensor_bps['collision'] = CollisionSensor.create_bp(self.params, world)
            elif sensor_type == 'RGBBirdsEyeView':
                self.sensor_bps['rgb_birds_eye_view'] = RGBBirdsEyeViewSensor.create_bp(self.params, world)
                self.rgb_bev_renderer = self.sensor_bps['rgb_birds_eye_view'][0](self.params, world)
            elif sensor_type == 'MultiBirdsEyeView':
                self.sensor_bps['multi_birds_eye_view'] = MultiBirdsEyeViewSensor.create_bp(self.params, world)
                self.multi_bev_renderer = self.sensor_bps['multi_birds_eye_view'][0](self.params, world)
            elif sensor_type == "SemanticLidar":
                self.sensor_bps['semantic_lidar'] = SemanticLidar.create_bp(self.params, world)
            elif sensor_type == "SemanticLidar360":
                self.sensor_bps['semantic_lidar360'] = SemanticLidar360.create_bp(self.params, world)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self._sensors.items():
            del s

    @property
    def sensors(self):
        return self._sensors

    def spawn_ego_sensors(self, ego):
        """Spawns sensors for the ego vehicle"""
        self.ego = ego
        for sensor_type in self.params.env.sensors.sensors:
            sensor = None
            if sensor_type == 'RGBCamera':
                sensor = RGBCamera('rgb_camera', self.params, ego,
                                   bp=self.sensor_bps['camera'][0],
                                   transform=self.sensor_bps['camera'][1])
            elif sensor_type == 'Lidar':
                sensor = Lidar('lidar', self.params, ego,
                               bp=self.sensor_bps['lidar'][0],
                               transform=self.sensor_bps['lidar'][1])
            elif sensor_type == 'SemanticLidar':
                sensor = SemanticLidar('semantic_lidar', self.params, ego,
                                       bp=self.sensor_bps['semantic_lidar'][0],
                                       transform=self.sensor_bps['semantic_lidar'][1])
            elif sensor_type == 'SemanticLidar360':
                sensor = SemanticLidar360('semantic_lidar360', self.params, ego,
                                          bp=self.sensor_bps['semantic_lidar360'][0],
                                          transform=self.sensor_bps['semantic_lidar360'][1])
            elif sensor_type == 'Collision':
                sensor = CollisionSensor('collision', self.params, ego,
                                         bp=self.sensor_bps['collision'][0],
                                         transform=self.sensor_bps['collision'][1])
            elif sensor_type == 'RGBBirdsEyeView':
                sensor = RGBBirdsEyeViewSensor('rgb_birds_eye_view', self.params,
                                               bev_renderer=self.rgb_bev_renderer,
                                               transform=None)
            elif sensor_type == 'MultiBirdsEyeView':
                sensor = MultiBirdsEyeViewSensor('multi_birds_eye_view', self.params,
                                                 bev_renderer=self.multi_bev_renderer,
                                                 transform=None)
            if sensor is not None:
                self.register(sensor)
                logging.debug(f'Spawned sensor {sensor.name}')
            else:
                raise Exception("Unknown sensor selected.")
        self.ego_traffic_light = None
        self.world.tick()

    def close_all_sensors(self):
        """Closes all sensors"""
        self.ego = None
        for sensor in self.sensors.values():
            sensor.close()
        self._sensor_queues = dict()
        self._event_sensor_queues = dict()

    def start_recorder(self, log_dir, name):
        """Starts the recorder for all sensors"""
        for sensor in self.sensors.values():
            if sensor.can_record:
                sensor.start_recorder(log_dir, name)

    def stop_recorder(self, rec_dir):
        """Stops the recorder for all sensors"""
        data = dict()
        for sensor in self.sensors.values():
            if sensor.can_record:
                # sensor.stop_recorder()
                data[sensor.name] = sensor.stop_recorder(rec_dir)
        return data

    def register(self, sensor):
        """Adds a specific sensor to the class"""
        self._sensors.update({sensor.name: sensor})

    def get_data(self, w_frame, waypoints=None):
        """Gets data from all sensors"""
        data_all = dict()
        for sensor_name, sensor in self._sensors.items():
            if isinstance(sensor, CustomCallSensor):
                if 'visible_actors' not in data_all.keys():
                    raise Exception("SemanticLidar sensor must be registered before BirdEyeView sensor.")
                # Call sensors are called here, e.g., the bird_evew_view sensor
                sensor.call(self.world, self.ego, waypoints, actor_list=data_all['visible_actors'][1])
                logging.debug(f'Call sensor {sensor_name} call.')

            t = time.time()
            while True:
                # Wait for normal sensors to fill something to their data queue; event sensors continue their execution
                if len(sensor.data_queue) > 0:
                    frame, data = sensor.data_queue.pop()
                    if frame == w_frame:
                        break
                    elif sensor.is_event_sensor:
                        frame = w_frame
                        break

                elif sensor.is_event_sensor:
                    logging.debug(f'Got data from {sensor_name} which is an event sensor.')
                    frame = w_frame
                    data = None
                    break

                elif time.time() - t > 1:
                    logging.debug(f'Sensor {sensor_name} timed out. Try again after waiting for some time!')
                    time.sleep(0.5)

            data_all.update({sensor_name: (frame, sensor.postprocess(data))})
            if sensor_name == 'semantic_lidar':
                # Use semantic lidar to get the visible actors
                data_all['visible_actors'] = (frame, self.get_visible_actors(data_all))

        return data_all

    def get_visible_actors(self, data):
        semantic_lidar = data['semantic_lidar'][1]
        actor_visible_ids_unfiltered = np.unique(semantic_lidar['ObjIdx'])

        all_actor_ids_existing_world = [actor.id for actor in self.world.get_actors()]
        all_actor_ids, vehicles_visible, walkers_visible, lights_visible, signs_visible = list(), list(), \
            list(), list(), list()

        for obj_idx in actor_visible_ids_unfiltered:
            if obj_idx in all_actor_ids_existing_world:
                obj_actor = self.world.get_actor(int(obj_idx))
                if obj_actor.type_id.startswith('vehicle'):
                    vehicles_visible.append(obj_actor)
                    all_actor_ids.append(int(obj_idx))
                elif obj_actor.type_id.startswith('walker'):
                    walkers_visible.append(obj_actor)
                    all_actor_ids.append(int(obj_idx))
                elif "traffic_light" in obj_actor.type_id:
                    lights_visible.append(obj_actor)
                elif 'stop' in obj_actor.type_id or 'yield' in obj_actor.type_id or 'speed_limit' in obj_actor.type_id:
                    signs_visible.append(obj_actor)
        return [vehicles_visible, walkers_visible, lights_visible, signs_visible, all_actor_ids]

    def collision_walker(self, data):
        collision = False

        # Use semantic lidar to detect collisions
        if 'SemanticLidar360' in self.params.env.sensors.sensors:
            semantic_lidar360 = data['semantic_lidar360'][1]
            actor_visible_ids_360_unfiltered = np.unique(semantic_lidar360['ObjIdx'])

            # Check collision with walker
            walkers = [self.world.get_actor(int(i)) for i in actor_visible_ids_360_unfiltered]
            if len(walkers) > 0:
                walkers_visible = [w for w in walkers if w is not None and w.type_id.startswith('walker')]
                if len(walkers_visible) > 0:
                    ego_bb = self.ego.bounding_box
                    # Extend the bounding box
                    ego_bb_extended = carla.BoundingBox(
                        ego_bb.location,
                        carla.Vector3D(ego_bb.extent.x + 0.35, ego_bb.extent.y + 0.35, ego_bb.extent.z + 0.5)
                    )
                    ego_verts = ego_bb_extended.get_world_vertices(self.ego.get_transform())

                    ego_verts = np.array([[ego_verts[0].x, ego_verts[0].y],
                                          [ego_verts[2].x, ego_verts[2].y],
                                          [ego_verts[4].x, ego_verts[4].y],
                                          [ego_verts[6].x, ego_verts[6].y]])

                    walker_verts = [w.bounding_box.get_world_vertices(w.get_transform()) for w in walkers_visible]

                    walker_verts = np.array([[[w[0].x, w[0].y],
                                              [w[2].x, w[2].y],
                                              [w[4].x, w[4].y],
                                              [w[6].x, w[6].y]] for w in walker_verts])
                    collision = collision_hero_to_multiple(ego_verts, walker_verts)
        else:
            print('semantic_lidar360 not in sensors')
        return collision

    def get_ego_state(self):
        # Get more data here about the ego vehicle
        # Position and yaw
        data = dict()
        ego_transform = self.ego.get_transform()
        data['ego_pos'] = np.array([ego_transform.location.x, ego_transform.location.y])
        data['ego_yaw'] = ego_transform.rotation.yaw / 180 * np.pi

        # Velocity
        ego_vel = self.ego.get_velocity()  # in m/s
        data['ego_vel'] = np.array([ego_vel.x, ego_vel.y])
        # Transform ego speed to ego frame
        R_to_ego = np.array([
            [np.cos(data['ego_yaw']), np.sin(data['ego_yaw'])],
            [-np.sin(data['ego_yaw']), np.cos(data['ego_yaw'])]
        ])
        data['ego_vel_norm'] = np.linalg.norm(data['ego_vel'])
        data['ego_vel_in_ego'] = R_to_ego.dot(np.array([ego_vel.x, ego_vel.y]))

        # Acceleration
        ego_acc = self.ego.get_acceleration()
        data['ego_acc'] = np.array([ego_acc.x, ego_acc.y])
        data['ego_acc_norm'] = np.linalg.norm(data['ego_acc'])
        data['ego_speed_limit'] = self.ego.get_speed_limit() / 3.6  # in m/s

        # Control
        ego_steer = self.ego.get_control().steer
        data['ego_steer'] = ego_steer

        # Speed tracking error
        speed_tracking_error = data['ego_vel_norm'] - data['ego_speed_limit']
        data['speed_tracking_error'] = data['ego_vel_norm'] - data['ego_speed_limit']
        data['speeding'] = speed_tracking_error > 0

        return data

    def run_red_light(self):
        run_red_light = False
        at_red_tfl = False
        if self.ego_traffic_light is not None:
            if self.ego_traffic_light.get_state() != carla.TrafficLightState.Red:
                self.ego_traffic_light = None
            else:
                at_red_tfl = True
                trigger_volume = self.ego_traffic_light.trigger_volume
                trigger_volume.extent = carla.Vector3D(trigger_volume.extent.x,
                                                       # Extend the trigger volume to the traffic light position
                                                       self.ego_traffic_light.trigger_volume.location.y,
                                                       # - self.ego_carla.bounding_box.extent.x,
                                                       trigger_volume.extent.z + 3)
                run_red_light = not trigger_volume.contains(self.ego.get_transform().location,
                                                            self.ego_traffic_light.get_transform())

                # # Draw bounding box of traffic light in carla
                # self.world.debug.draw_box(
                # carla.BoundingBox(self.ego_traffic_light.get_transform().location,
                # self.ego_traffic_light.bounding_box.extent),
                # self.ego_traffic_light.get_transform().rotation, 0.1, carla.Color(255, 0, 0, 0))
                #
                # # Draw bounding box of the extended trigger volume
                # traffic_light_volume_location = self.ego_traffic_light.get_transform().transform(
                # trigger_volume.location)
                # traffic_light_volume_extent = carla.Vector3D(trigger_volume.extent.x,
                # self.ego_traffic_light.trigger_volume.location.y,
                # trigger_volume.extent.z)
                #
                # self.world.debug.draw_box(carla.BoundingBox(traffic_light_volume_location, traffic_light_volume_extent),
                # self.ego_traffic_light.get_transform().rotation, 0.1,
                # carla.Color(255, 0, 0, 0))

                if run_red_light:  # and self.ego_carla.get_traffic_light_state() == carla.TrafficLightState.Red:
                    self.ego_traffic_light = None
        elif self.ego.is_at_traffic_light():
            get_traffic_light_state = self.ego.get_traffic_light_state()
            if get_traffic_light_state == carla.TrafficLightState.Red:
                at_red_tfl = True
                self.ego_traffic_light = self.ego.get_traffic_light()
            # else:
            #     traffic_light_near = True
        return run_red_light, at_red_tfl

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self._sensors.items():
            del s
        try:
            del self.pygame_recorder
            del self.rgb_bev_renderer
            del self.multi_bev_renderer
        except:
            pass

        super().__exit__(exc_type, exc_val, exc_tb)
