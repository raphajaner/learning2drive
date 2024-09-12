from envs.carla_gym.actors.actors import *
from numpy import random
import logging
import carla
from envs.carla_gym.misc import calc_lateral_distance, calc_longitudinal_distance

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


class ActorManager:
    """ Class to manage actors in the simulation """

    def __init__(self, params, client):
        self.params = params
        self.client = client
        self.world = self.client.get_world()
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())

        # Keeping track of all actors, sensors, etc. in the simulation
        self.ego = None
        self.vehicles_id_list = []
        self.walkers_id_list = []
        self.walker_controllers_id_list = []
        self.all_id_list = []

        # Allocate bps
        # Ego
        self.ego_bp = create_ego_bp(params, self.world)
        # Cars
        self.vehicle_bps = [create_vehicle_bp(self.world) for _ in range(200)]
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # Pedestrians
        self.walker_bps = [create_walker_bp(self.world) for _ in range(100)]
        self.walker_controller_bp = create_walker_controller_bp(self.world)
        # Preallocate Transform for pedestrian navigation
        self.walker_spawn_points = [carla.Transform() for _ in range(5000)]
        valid_walker_spawn_points = []
        for sp in self.walker_spawn_points:
            while True:
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    sp.location = loc
                    valid_walker_spawn_points.append(sp)
                    break
        self.walker_spawn_points = valid_walker_spawn_points

    def spawn_ego(self, params):
        """ Spawn the ego vehicle """
        if self.ego is not None and self.ego.is_alive:
            self.logger.info(
                "Ego vehicle already exists and is alive. "
                "Please make sure that the ego is correctly deleted before spawning")

        random.shuffle(self.vehicle_spawn_points)
        for i in range(0, len(self.vehicle_spawn_points)):
            logging.debug(f'Trying to spawn ego: attempt {i}.')
            next_spawn_point = self.vehicle_spawn_points[i]
            overlap = False
            for actor in self.world.get_actors().filter('vehicle*'):
                actor_loc = actor.get_location()
                actor_x = actor_loc.x
                actor_y = actor_loc.y
                spawn_point_loc = next_spawn_point.location
                spawn_point_rot = np.deg2rad(next_spawn_point.rotation.yaw)
                spawn_point_x = spawn_point_loc.x
                spawn_point_y = spawn_point_loc.y
                if np.abs(calc_lateral_distance(actor_x, actor_y, spawn_point_rot, spawn_point_x,
                                                spawn_point_y)) < 1.5 and \
                        - 20 < calc_longitudinal_distance(actor_x, actor_y, spawn_point_rot, spawn_point_x,
                                                          spawn_point_y) < 30.0:
                    overlap = True
                    break

            if not overlap:
                self.ego = self.world.try_spawn_actor(self.ego_bp, next_spawn_point)
                if self.ego is not None:
                    self.logger.debug("Ego spawned!")
                    self.all_id_list.append(self.ego.id)
                    return
                else:
                    self.logger.warning("Could not spawn hero, changing spawn point")

        raise Exception("We ran out of spawn points")

    def spawn_vehicles(self, n_vehicles, tm_port):
        """ Spawn n_vehicles in the simulation """
        batch_vehicle = []
        np.random.shuffle(self.vehicle_spawn_points)

        # Crate a batch to spawn all n_vehicles synchronously
        for n, transform in enumerate(self.vehicle_spawn_points):
            if n >= n_vehicles:
                break
            blueprint = np.random.choice(self.vehicle_bps)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # Spawn the cars and set their autopilot and light state all together
            batch_vehicle.append(SpawnActor(blueprint, transform)
                                 .then(SetAutopilot(FutureActor, True, tm_port)))

        # Spawn as batch
        response = self.client.apply_batch_sync(batch_vehicle, False)
        for results in response:
            if results.error:
                self.logger.error(f"Spawning vehicles lead to error: {results.error}")
            else:
                self.vehicles_id_list.append(results.actor_id)

        self.logger.debug(f'Spawned {len(self.vehicles_id_list)} vehicles.')
        self.all_id_list += self.vehicles_id_list

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def spawn_walkers(self, n_walkers):
        """ Spawn n_walkers in the simulation """
        self.world.set_pedestrians_seed(self.params.env.other_road_users.pedestrian_seed)

        walkers_list = []
        walkers_list += self._spawn_batch_walkers(n_walker=n_walkers - len(walkers_list))

        self.walkers_id_list = walkers_list
        self.all_id_list += self.walkers_id_list

        self.logger.debug(f'Spawned {len(self.walkers_id_list)} walkers.')

        # Spawn the walker controller
        self.spawn_walker_controllers(self.walkers_id_list)

    def _spawn_batch_walkers(self, n_walker):
        """ Spawn n_walker walkers in the simulation """
        # We spawn the walker object
        walker_bps = np.random.choice(self.walker_bps, n_walker, replace=True)
        walker_spawn_points = np.random.choice(self.walker_spawn_points, n_walker, replace=False)
        walkers_list = []
        batch = [SpawnActor(bp, sp) for (bp, sp) in zip(walker_bps, walker_spawn_points)]

        response = self.client.apply_batch_sync(batch, False)

        for result in response:
            if result.error:
                self.logger.debug(f"Spawning walkers lead to error: {result.error}")
            else:
                walkers_list.append(result.actor_id)

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        return walkers_list

    def spawn_walker_controllers(self, walker_id_list):
        """ Spawn the walker controllers """
        batch = [SpawnActor(self.walker_controller_bp, carla.Transform(), walker_id) for walker_id in walker_id_list]

        response = self.client.apply_batch_sync(batch, False)

        for result in response:
            if result.error:
                self.logger.error(f"Error when spawning the controllers: {result.error}")
            else:
                self.walker_controllers_id_list.append(result.actor_id)

        self.logger.debug(f'Spawned {len(self.walker_controllers_id_list)} walker controllers.')

        self.all_id_list += self.walker_controllers_id_list

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        walker_controllers_list = self.world.get_actors(self.walker_controllers_id_list)

        # Start the controllers
        for walker_controller in walker_controllers_list:
            walker_controller.start()
            target_locations = np.random.choice(self.walker_spawn_points).location
            walker_controller.go_to_location(target_locations)
            # max speed
        self.logger.debug(f'Started all pedestrian controllers.')

    def clear_batch(self, id_list, type='actor'):
        """ Destroy the actors in the id_list """
        synchronous_mode = self.world.get_settings().synchronous_mode

        if isinstance(id_list, list):
            id_list = [DestroyActor(x) for x in id_list]
        else:
            id_list = [DestroyActor(id_list)]

        response = self.client.apply_batch_sync(id_list)
        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    self.logger.error(f"A {type} could not be destroyed: {result.error}.")
                else:
                    n_deleted += 1
            self.logger.debug(f'Destroyed {n_deleted} {type}(s).')
        else:
            self.logger.error(f"There were no {type}(s) to be destroyed.")

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def clear_vehicles(self):
        """ Destroy all vehicles """
        self.clear_batch(self.vehicles_id_list, 'vehicle')
        self.all_id_list = [actor_id for actor_id in self.all_id_list
                            if actor_id not in self.vehicles_id_list]
        self.vehicles_id_list = []

    def clear_walker(self):
        """ Destroy all walkers """
        for walker_controller in self.world.get_actors(self.walker_controllers_id_list):
            walker_controller.stop()

        self.clear_batch(self.walker_controllers_id_list, 'controller')
        self.clear_batch(self.walkers_id_list, 'walker')
        self.all_id_list = [actor_id for actor_id in self.all_id_list
                            if actor_id not in self.walkers_id_list
                            and actor_id not in self.walker_controllers_id_list]
        self.walkers_id_list = []
        self.walker_controllers_id_list = []

    def clear_ego(self):
        """ Destroy the ego vehicle """
        self.clear_batch(self.ego.id, 'ego vehicle')
        self.all_id_list = [actor_id for actor_id in self.all_id_list
                            if actor_id is not self.ego.id]
        self.ego = None

    def clear_all_actors(self):
        """ Destroy all actors """
        self.clear_ego()
        self.clear_vehicles()
        self.clear_walkers()
        self.logger.debug(f"Actors have been destroyed.")
