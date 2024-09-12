import pdb
import time
from abc import ABC, abstractmethod
import numpy as np
import carla
import logging
from queue import Queue, Empty
from collections import deque
from threading import Thread
from numba import njit, jit
from numba.experimental import jitclass
from envs.carla_gym.sensors.bird_eye_view_sensor_cv2 import RGBBirdsEyeViewRenderer, MultiBirdsEyeViewRenderer


class BaseSensor(ABC):
    def __init__(self, name, params, parent):
        # Use a queue for thread safe access to the data
        # Note deque is also thread safe! See: https://realpython.com/python-deque/
        self.data_queue = deque(maxlen=1)  # deque(maxlen=5)

        self.name = name
        self.params = params
        self.sensor = None
        self.is_event_sensor = False
        self.can_record = False

    @property
    def id(self):
        return self.sensor.id

    def parse(self, data):
        return data

    def update(self, frame, data):
        data_processed = self.parse(data)
        self.data_queue.append((frame, data_processed))

    def callback(self, data):
        """ The callback is wrapping the update function to allow the use in the sensor.listen function.
        Otherwise, a lambda function had to be used.
        """
        self.update(data.frame, data)

    def postprocess(self, data):
        return data

    def close(self):
        if self.sensor is not None:
            if hasattr(self.sensor, 'is_listening') and self.sensor.is_listening:
                self.sensor.stop()
            self.sensor.destroy()
            del self.sensor


class CarlaSensor(BaseSensor):
    def __init__(self, name, params, parent, bp, transform):
        # Use a queue for thread safe access to the data
        super().__init__(name, params, parent)
        world = parent.get_world()
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent)
        self.sensor.listen(self.callback)

    @staticmethod
    def create_bp(params, world):
        pass


class CustomThreadSensor(BaseSensor):
    def __init__(self, name, params, parent, bp, target_func):
        super().__init__(name, params, parent)
        self.running = False
        self.previous_frame = None
        self.sensor = bp(params, parent)

        self.thread = Thread(target=target_func, args=(parent.get_world(), parent))
        self.thread.daemon = True
        self.thread.start()

    def callback(self, frame, data):
        # Difference to normal sensor is that callback takes data and frame instead of just data
        self.update(frame, data)

    def close(self):
        """Stop the thread + sensor and its execution"""
        self.running = False
        self.thread.join()
        del self.thread
        self.sensor.destroy()


class CustomCallSensor(BaseSensor):
    def __init__(self, name, params, parent, bp):
        super().__init__(name, params, parent)
        self.running = False
        self.previous_frame = None

    def call(self, world, egp):
        pass

    def callback(self, frame, data):
        # Difference to normal sensor is that callback takes data and frame instead of just data
        self.update(frame, data)


class EventSensor(CarlaSensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_event_sensor = True
        self.last_event_frame = None


class ContinuousSensor(CarlaSensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_event_sensor = False


class RGBCamera(ContinuousSensor):
    def __init__(self, name, params, parent, bp, transform):
        super().__init__(name, params, parent, bp, transform)

    @staticmethod
    def create_bp(params, world):
        obs_size = int(params.env.sensors.obs_range / params.env.sensors.lidar_bin)
        transform = carla.Transform(carla.Location(x=0.8, z=1.7))

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(obs_size))
        bp.set_attribute('image_size_y', str(obs_size))
        bp.set_attribute('fov', str(params.env.sensors.fov))
        bp.set_attribute('enable_postprocess_effects', 'false')
        # Set the time in seconds between sensor captures
        bp.set_attribute('sensor_tick', '0.01')
        return bp, transform

    def parse(self, data):
        data_processed = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        data_processed = np.reshape(data_processed, (data.height, data.width, 4))
        data_processed = data_processed[:, :, :3]
        return data_processed[:, :, ::-1]


class Lidar(ContinuousSensor):
    def __init__(self, name, params, parent, bp, transform):
        super().__init__(name, params, parent, bp, transform)
        self.obs_size = int(params.env.sensors.obs_range / params.env.sensors.lidar_bin)
        self.lidar_height = params.env.sensors.lidar_height
        # Separate the 3D space to bins for point cloud, x and y is set according to self.params['lidar_bin'],
        # and z is set to be two bins.
        self.y_bins = np.arange(-(self.params.env.sensors.obs_range - self.params.env.sensors.d_behind),
                                self.params.env.sensors.d_behind + self.params.env.sensors.lidar_bin,
                                self.params.env.sensors.lidar_bin)
        self.x_bins = np.arange(-self.params.env.sensors.obs_range / 2,
                                self.params.env.sensors.obs_range / 2 + self.params.env.sensors.lidar_bin,
                                self.params.env.sensors.lidar_bin)
        self.z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]

    @staticmethod
    def create_bp(params, world):
        transform = carla.Transform(carla.Location(x=0.0, z=params.env.sensors.lidar_height))
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '32')
        # range	float	10.0
        # Maximum distance to measure/raycast in meters (centimeters for CARLA 0.9.6 or previous).
        bp.set_attribute('range', str(params.env.sensors.obs_range))
        return bp, transform

    def postprocess(self, data):
        point_cloud = np.array([[d.point.x, d.point.y, -d.point.z] for d in data])
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(self.x_bins, self.y_bins, self.z_bins))
        lidar[:, :, 0] = np.greater(lidar[:, :, 0], 0)
        lidar[:, :, 1] = np.greater(lidar[:, :, 1], 0)

        # Need to add a third dim (blue channel of RGB) for image
        blue_dim = np.zeros(lidar.shape[:2])
        blue_dim = np.expand_dims(blue_dim, axis=2)
        lidar = np.concatenate((lidar, blue_dim), axis=2)
        lidar = np.fliplr(lidar)
        lidar = np.flipud(lidar)
        return lidar.astype(np.uint8)

    def add_waypoints_to_image(self, image, birdeye):
        if self.params.env.sensors.display_route:
            wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
        else:
            wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))
        return image


class SemanticLidar(ContinuousSensor):
    def __init__(self, name, params, parent, bp, transform):
        super().__init__(name, params, parent, bp, transform)

    @staticmethod
    def create_bp(params, world):
        transform = carla.Transform(carla.Location(x=0.0, z=params.env.sensors.lidar_height))
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        bp.set_attribute('range', str(params.env.sensors.obs_range))
        bp.set_attribute('horizontal_fov', '110.0')
        bp.set_attribute('rotation_frequency', '10.0')

        return bp, transform

    def parse(self, data):
        return data

    def postprocess(self, data):
        sem_lidar_dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                                 ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])
        out = np.frombuffer(data.raw_data, dtype=sem_lidar_dt)
        return out


class SemanticLidar360(SemanticLidar):

    @staticmethod
    def create_bp(params, world):
        transform = carla.Transform(carla.Location(x=0.0, z=params.env.sensors.lidar_height))
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        bp.set_attribute('range', '10')
        bp.set_attribute('upper_fov', '50.0')
        bp.set_attribute('lower_fov', '-50.0')
        bp.set_attribute('horizontal_fov', '360.0')
        bp.set_attribute('points_per_second', '100000')
        bp.set_attribute('rotation_frequency', '10.0')

        return bp, transform


class CollisionSensor(EventSensor):
    def __init__(self, name, params, parent, bp, transform):
        super().__init__(name, params, parent, bp, transform)

    @staticmethod
    def create_bp(params, world):
        bp = world.get_blueprint_library().find('sensor.other.collision')
        return bp, carla.Transform()

    def parse(self, data):
        impulse = data.normal_impulse
        intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        return [data.other_actor, intensity]

    def callback(self, data):
        self.update(data.frame, data)


class RGBBirdsEyeViewSensor(CustomCallSensor):
    """
    This class is responsible for creating a 'birdview' pseudo-sensor, which is a simplified
    version of CARLA's non rendering mode.
    """

    def __init__(self, name, params, bev_renderer, transform=None):
        super().__init__(name, params, None, None)
        self.sensor = bev_renderer
        self.can_record = True
        self.log_dir = None

    @staticmethod
    def create_bp(params, world):
        return RGBBirdsEyeViewRenderer, None

    def call(self, world, ego, waypoints, actor_list):
        frame = world.get_snapshot().frame
        data = self.sensor.get_data(world, ego, waypoints, actor_list)
        self.callback(frame, data)
        self.previous_frame = frame

    def start_recorder(self, log_dir, num):
        pass

    def stop_recorder(self, rec_dir):
        return None


class MultiBirdsEyeViewSensor(CustomCallSensor):
    """
    This class is responsible for creating a 'birdview' pseudo-sensor, which is a simplified
    version of CARLA's non rendering mode.
    """

    def __init__(self, name, params, bev_renderer, transform=None):
        super().__init__(name, params, None, None)
        self.sensor = bev_renderer
        self.can_record = True
        self.log_dir = None

    @staticmethod
    def create_bp(params, world):
        return MultiBirdsEyeViewRenderer, None

    def call(self, world, ego, waypoints, actor_list):
        frame = world.get_snapshot().frame
        data = self.sensor.get_data(world, ego, waypoints, actor_list)
        self.callback(frame, data)
        self.previous_frame = frame

    def start_recorder(self, log_dir, num):
        pass

    def stop_recorder(self, rec_dir):
        return None
