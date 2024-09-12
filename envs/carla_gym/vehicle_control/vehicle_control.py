import numpy as np
import carla
from envs.carla_gym.vehicle_control.route_planner import RoutePlanner
from envs.carla_gym.vehicle_control.route_planner import RoadOption


class VehicleController:
    def __init__(self, params, ego_carla):
        self.params = params
        self.ego_carla = ego_carla

        self.route_planner = RoutePlanner(self.ego_carla, self.params.env.ego.max_waypt,
                                          self.params.env.random_road_options)
        self.waypoints = None
        self.vehicle_front = None
        self.target_speed = None

        self.lateral_controller = None
        self.longitudinal_controller = None

        self.state = {}
        self.last_action = None

    def update_state(self, snapshot=None):
        if snapshot is not None:
            ego_carla = snapshot.find(self.ego_carla.id)
        else:
            ego_carla = self.ego_carla

        ego_transform = ego_carla.get_transform()
        self.state['location'] = [ego_transform.location.x, ego_transform.location.y]
        # Save yaw in radians (rotation.yaw is given in degree)
        self.state['yaw'] = np.math.radians(ego_transform.rotation.yaw)
        ego_velocity = ego_carla.get_velocity()
        self.state['velocity_2d'] = [ego_velocity.x, ego_velocity.y]
        self.state['velocity_2d_abs'] = np.linalg.norm([ego_velocity.x, ego_velocity.y])
        self.state['speed_limit'] = self.ego_carla.get_speed_limit() / 3.6
        ego_acc = ego_carla.get_acceleration()
        self.state['acc_2d'] = [ego_acc.x, ego_acc.y]
        self.state['acc_2d_abs'] = np.linalg.norm([ego_acc.x, ego_acc.y])

    def update_route(self):
        self.waypoints, _, _ = self.route_planner.run_step()
        self.target_speed = self.ego_carla.get_speed_limit() / 3.6
        # logging.info(f'Current speed limit: {self.target_speed:0.2f}m/s.')

    def get_target_waypoint(self):
        return self.route_planner.get_target_waypoint()

    def get_control_action(self, acceleration=None, steer=None):
        if acceleration is None:
            if self.longitudinal_controller is not None:
                acceleration = self.longitudinal_controller.get_control(self.state, self.target_speed)
            else:
                raise Exception('Longitudinal drive dynamics controller is missing!')

        # Convert acceleration to throttle and brake and scale from [-1, 1] to control
        if acceleration > 0:
            throttle = np.clip(acceleration, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acceleration, 0, 1)

        if steer is None:
            if self.lateral_controller is not None:
                steer = self.lateral_controller.get_control(self.state, self.waypoints)
            else:
                raise Exception('Lateral drive dynamics controller is missing!')

        # logging.info(f'Ego action: throttle={throttle:0.3f}, brake={brake:0.3f}, steer={steer:0.3f}')

        return throttle, brake, steer

    def get_carla_control_action(self, acceleration=None, steer=None):
        throttle, brake, steer = self.get_control_action(acceleration, steer)
        action = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        return action

    def apply_carla_control_action(self, action):
        self.ego_carla.apply_control(action)
        self.last_action = action

    def add_lateral_controller(self, controller_cls):
        self.lateral_controller = controller_cls(self.ego_carla)

    def add_longitudinal_controller(self, controller_cls):
        self.longitudinal_controller = controller_cls(self.ego_carla)

    def affected_by_other_agents(self):
        pass

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance
