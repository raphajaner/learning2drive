import logging

import carla
import numpy as np
import math
from envs.carla_gym.misc import distance_vehicle, distance_vehicle_no_transform_wp
from abc import ABC, abstractmethod
from envs.carla_gym.misc import normalize_angle
from queue import Queue
from collections import deque


class DriveDynamicsController(ABC):
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.offset_length = self.calc_front_axle()

    def calc_front_axle(self):
        physics = self.vehicle.get_physics_control()
        wheels = physics.wheels
        # Position returns a Vector3D with measures in cm
        wheel_FL = wheels[0].position
        wheel_FR = wheels[1].position
        ego_location = self.vehicle.get_transform().location  # Vector3D(x=10, y=20 z=0.5)
        offset_vec_FL = carla.Vector3D(wheel_FL.x / 100, wheel_FL.y / 100, wheel_FL.z / 100).__sub__(ego_location)
        offset_vec_FR = carla.Vector3D(wheel_FR.x / 100, wheel_FR.y / 100, wheel_FR.z / 100).__sub__(ego_location)
        offset_vec = offset_vec_FL + offset_vec_FR
        offset_length = np.sqrt(offset_vec.x ** 2 + offset_vec.y ** 2) / 2
        return offset_length

    def get_front_axle_position(self):
        ego_transform = self.vehicle.get_transform()
        ego_location_x_front_axle = ego_transform.location.x + self.offset_length * math.cos(
            math.radians(ego_transform.rotation.yaw))
        ego_location_y_front_axle = ego_transform.location.y + self.offset_length * math.sin(
            math.radians(ego_transform.rotation.yaw))
        offset = carla.Transform(carla.Location(ego_location_x_front_axle, ego_location_y_front_axle),
                                 ego_transform.rotation)
        ego_f_v = ego_transform.get_forward_vector()
        front_axle_pos = carla.Transform(ego_transform.location.__add__(ego_f_v.__mul__(self.offset_length)),
                                         ego_transform.rotation)
        return front_axle_pos

    def get_target_waypoint(self, waypoints):
        wp_ind = np.argmin([distance_vehicle_no_transform_wp(wp, self.get_front_axle_position()) for wp in waypoints])
        wp_target = waypoints[wp_ind]
        return wp_target

    @abstractmethod
    def get_control(self, state, waypoints):
        pass


class StanleyLateralVehicleController(DriveDynamicsController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.k_e = 1
        self.k_v = 10

        self.yaw_previous = None
        self.wp_target_previous = None

    def get_control(self, state, waypoints):
        """
            Stanley controller for lateral vehicle vehicle_control. Derived from:
            https://github.com/diegoavillegasg/Longitudinal-and-Lateral-Controller-on-Carla/blob/master/controller2d.py
                position        : Current [x, y] position in meters
                velocity        : Current forward speed (meters per second)
                yaw             : Current yaw pose in radians
                waypoints       : Current waypoints to track
                                  (Includes speed to track at each x,y
                                  location.)
                                  Format: [[x0, y0, v0],
                                           [x1, y1, v1],
                                           ...
                                           [xn, yn, vn]]
                                  Example:
                                      waypoints[2][1]:
                                      Returns the 3rd waypoint's y position
                                      waypoints[5]:
                                      Returns [x5, y5, v5] (6th waypoint)
        """

        # Change the steer output with the lateral controller.
        steer_output = 0

        position = state['location']
        velocity = state['velocity_2d_abs']
        yaw = state['yaw']

        # Get waypoint that is nearest to front wheel
        self.front_axle_position = self.get_front_axle_position()
        wp_ind = np.argmin([distance_vehicle_no_transform_wp(wp, self.front_axle_position) for wp in waypoints])
        wp_target = waypoints[wp_ind]
        yaw_path = math.radians(wp_target[2])

        yaw_path = normalize_angle(yaw_path)
        yaw = normalize_angle(yaw)

        if self.yaw_previous is None:
            self.yaw_previous = yaw

        if self.wp_target_previous is None:
            self.wp_target_previous = wp_target

        # Heading error of car to trajectory
        yaw_diff = (yaw_path - yaw)
        yaw_diff_norm = normalize_angle(yaw_diff)
        front_axle_position = self.get_front_axle_position().location
        front_axle_vec = [front_axle_position.x, front_axle_position.y]

        front_axle_vec_norm = front_axle_vec / np.linalg.norm(front_axle_vec)
        a1_projection = np.dot(wp_target[:2], front_axle_vec_norm) * front_axle_vec_norm
        crosstrack_error = np.linalg.norm(wp_target[:2] - a1_projection)

        yaw_cross_track = np.arctan2(wp_target[1] - self.get_front_axle_position().location.y,
                                     wp_target[0] - self.get_front_axle_position().location.x)
        yaw_cross_track = normalize_angle(yaw_cross_track)
        yaw_path2ct = normalize_angle(yaw_path - yaw_cross_track)

        if yaw_path2ct < 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(self.k_e * crosstrack_error / (self.k_v + velocity))

        # Yaw daping for extended Stanley vehicle_control.
        # yaw_rate_trajectory = normalize_angle(wp_target[2] - self.wp_target_previous[2])
        # yaw_rate_ego = normalize_angle(yaw - self.yaw_previous)
        # yaw_rate_damping = 0 * (yaw_rate_ego - yaw_rate_trajectory)

        # 3. vehicle_control low
        steer_expect = 1.5 * yaw_diff_norm + yaw_diff_crosstrack  # + yaw_rate_damping

        if abs(steer_expect) < np.radians(1):
            steer_expect = steer_expect

        # 4. update
        steer_output = - steer_expect

        input_steer = (180.0 / np.pi) * steer_output / 69.999  # Max steering angle
        steer = np.clip(input_steer, -1.0, 1.0)
        self.yaw_previous = yaw
        self.wp_target_previous = wp_target

        return steer


class PIDLongitudinalVehicleController(DriveDynamicsController):
    """
    PIDLongitudinalController implements longitudinal vehicle_control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.1):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)
        super().__init__(vehicle)

    def get_control(self, state, target_speed):
        """
        Execute one step of longitudinal vehicle_control to reach a given target speed.
            :param state: state of the vehicle
            :param target_speed: target speed in Km/h
            :return: throttle vehicle_control
        """
        current_speed = state['velocity_2d_abs']

        logging.info('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake vehicle_control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)
        logging.info('Current speed error = {}'.format(error))

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -8.0, 3.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLongitudinalAccelerationVehicleController(PIDLongitudinalVehicleController):
    """PIDLongitudinalController implements longitudinal vehicle_control using a PID.
    """

    def get_control(self, state, target_acc):
        """
        Execute one step of longitudinal vehicle_control to reach a given target speed.
            :param state: state of the vehicle
            :param target_speed: target speed in Km/h
            :return: throttle vehicle_control
        """
        current_acc = state['acc_2d_abs']

        target_acc = 5 * target_acc  # scale range from [-1, 1] to [-5, 5] m/s^2

        logging.info(f'Current acc = {current_acc}, target_acc = {target_acc}')

        return self._pid_control(target_acc, current_acc)


class PIDLateralVehicleController(DriveDynamicsController):
    """
    PIDLateralController implements lateral vehicle_control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.1):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)
        super().__init__(vehicle)

    def get_control(self, state, waypoints):
        """
        Execute one step of lateral vehicle_control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoints
            :return: steering vehicle_control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(state, self.get_target_waypoint(waypoints))

    def _pid_control(self, state, waypoint):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering vehicle_control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = state['location']
        v_vec = state['velocity_2d']
        v_vec = np.array([v_vec[0], v_vec[1], 0.0])

        waypoint = carla.Transform(carla.Location(waypoint[0], waypoint[1], 0), carla.Rotation(waypoint[2]))

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset * r_vec.x,
                                                     y=self._offset * r_vec.y)
        else:
            w_loc = waypoint.location

        w_vec = np.array([w_loc.x - ego_loc[0],
                          w_loc.y - ego_loc[1],
                          0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            try:
                _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
            except:
                raise ValueError(
                    "math domain error in acos: norm is {}, dot is {}".format(wv_linalg, np.dot(w_vec, v_vec)))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return -np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
