# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import cv2
import numpy as np
import carla

from envs.carla_gym.sensors.color_keys import *
from envs.carla_gym.sensors.map_image import MapImage


class BirdsEyeViewRenderer:
    """Class that contains all the information of the carla world (in the form of pygame surfaces)"""

    def __init__(self, params, world):
        self.params = params
        self.scale_factor = np.sqrt(2)
        self.size = int(np.ceil(np.sqrt(2) * self.params.env.sensors.size_output_image))

        # Correct scaling for the offset
        self.radius = params.env.sensors.obs_range / 2
        self.pixels_per_meter = self.params.env.sensors.size_output_image / (2 * self.radius)
        # The math.sqrt(2) is a patch due to the later rotation and zoom of this image
        self.map_image = MapImage(world, world.get_map(), self.pixels_per_meter)  # / self.scale_factor)

        self.map_surface = pygame.surfarray.array3d(self.map_image.surface).swapaxes(0, 1)
        self.surface = None
        self.surface_size = None
        self.rotate_interpolate = cv2.INTER_AREA

    def _split_actors(self, world, actor_list):
        """Splits the retrieved actors by type id"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []
        if actor_list is None:
            for actor in world.get_actors():
                if 'vehicle' in actor.type_id:
                    vehicles.append(actor)
                elif 'walker.pedestrian' in actor.type_id:
                    walkers.append((actor))
                elif 'traffic_light' in actor.type_id:
                    traffic_lights.append(actor)
                elif 'speed_limit' in actor.type_id:
                    speed_limits.append(actor)
        else:
            vehicles_visible, walkers_visible, lights_visible, signs_visible, all_actor_ids, hero = actor_list
            vehicles = vehicles_visible + [hero]
            walkers = walkers_visible
            speed_limits = signs_visible
            traffic_lights = lights_visible
        return vehicles, traffic_lights, speed_limits, walkers

    def _render_map(self, surface, offset):
        raise NotImplementedError

    def _render_waypoints(self, surface, waypoints, offset):
        raise NotImplementedError

    def _render_traffic_lights(self, surface, traffic_lights, offset):
        raise NotImplementedError

    def _render_vehicles(self, surface, list_v, offset, surface_vel=None, ego=None):
        raise NotImplementedError

    def _render_walkers(self, surface: np.array, list_w, offset, surface_vel=None):
        raise NotImplementedError

    def get_data(self, world, hero, waypoints, actor_list=None):
        """Renders the map and all the actors in hero and map mode"""
        hero_transform = hero.get_transform()
        hero_center_location = hero_transform.location + hero_transform.get_forward_vector() * self.radius / 2
        hero_screen_location = self.map_image.world_to_pixel(hero_center_location)
        angle = hero_transform.rotation.yaw + 90.0
        offset = np.array([hero_screen_location[0] - self.size / 2, hero_screen_location[1] - self.size / 2],
                          dtype=np.int32)
        surface = self.surface
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors(world, actor_list=actor_list + [hero])

        # RENDERING
        surface = self._render_map(surface, offset)
        if waypoints is not None:
            self._render_waypoints(surface, waypoints, offset)
        surface = self._render_traffic_lights(surface, traffic_lights, offset)
        surface = self._render_vehicles(surface, vehicles, offset, ego=hero)
        surface = self._render_walkers(surface, walkers, offset)

        # ROTATION
        image_center = np.array(surface.shape[1::-1]) / 2
        rot_mat = cv2.getRotationMatrix2D((image_center[0], image_center[1]), angle, 1)
        image = cv2.warpAffine(surface, rot_mat, surface.shape[1::-1], flags=self.rotate_interpolate)

        # get the center of the image
        image_center = tuple(np.array(image.shape[1::-1]) // 2)
        # select (n, n) region around the center of the image
        size_output_image = self.params.env.sensors.size_output_image
        image = image[image_center[0] - size_output_image // 2:image_center[0] + size_output_image // 2,
                image_center[1] - size_output_image // 2:   image_center[1] + size_output_image // 2]

        # import matplotlib.pyplot as plt
        # for i in range(image.shape[-1]):
        #     # makedir
        #     if not os.path.exists('images'):
        #         os.makedirs('images')
        #     plt.imshow(image[:, :, i], cmap='gray', vmin=0, vmax=1)
        #     plt.savefig(f'images/channel_{i}.png')
        #     plt.close()

        return image

    def destroy(self):
        """Destroy the hero actor when class instance is destroyed"""
        pass  # del self.map_image


class RGBBirdsEyeViewRenderer(BirdsEyeViewRenderer):
    """Class that contains all the information of the carla world (in the form of pygame surfaces)"""

    def __init__(self, params, world):
        # world = hero.get_world()
        super().__init__(params, world)
        self.surface = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.surface_size = (np.shape(self.surface)[0], np.shape(self.surface)[1])

    def _split_actors(self, world, actor_list):
        """Splits the retrieved actors by type id"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []
        if actor_list is None:
            for actor in world.get_actors():
                if 'vehicle' in actor.type_id:
                    vehicles.append(actor)
                elif 'walker.pedestrian' in actor.type_id:
                    walkers.append((actor))
                elif 'traffic_light' in actor.type_id:
                    traffic_lights.append(actor)
                elif 'speed_limit' in actor.type_id:
                    speed_limits.append(actor)
        else:
            vehicles_visible, walkers_visible, lights_visible, signs_visible, all_actor_ids, hero = actor_list
            vehicles = vehicles_visible + [hero]
            walkers = walkers_visible
            speed_limits = signs_visible
            traffic_lights = lights_visible
        return vehicles, traffic_lights, speed_limits, walkers

    def _render_map(self, surface, offset):
        # Reset the surface
        np.copyto(
            surface,
            self.map_surface[int(offset[1]): int(offset[1] + self.size), int(offset[0]): int(offset[0]) + self.size, :]
        )
        return surface

    def _render_waypoints(self, surface, waypoints, offset):
        # purple
        color = COLOR_PURPLE
        corners = [self.map_image.world_to_pixel(carla.Location(x=p[0], y=p[1]), offset=offset) for p in waypoints]
        radius = self.map_image.world_to_pixel_width(0.6)
        surface = cv2.polylines(surface, np.int32([corners]), False, (color.r, color.g, color.b), radius)
        return surface

    def _render_traffic_lights(self, surface, traffic_lights, offset):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled."""
        for tl in traffic_lights:
            pos = self.map_image.world_to_pixel(tl.get_location(), offset=offset)
            radius = self.map_image.world_to_pixel_width(2)  # 1.4)
            if tl.state == carla.TrafficLightState.Red:
                color = COLOR_SCARLET_RED_0
            elif tl.state == carla.TrafficLightState.Yellow:
                color = COLOR_BUTTER_0
            elif tl.state == carla.TrafficLightState.Green:
                color = COLOR_CHAMELEON_0
            elif tl.state == carla.TrafficLightState.Off:
                color = COLOR_ALUMINIUM_4
            else:
                color = COLOR_BLACK
            surface = cv2.circle(surface, (pos[0], pos[1]), radius, (color.r, color.g, color.b), cv2.FILLED)
        return surface

    def _render_walkers(self, surface: np.array, list_w, offset, surface_vel=None):
        """Renders the walkers' bounding boxes"""
        for w in list_w:
            color = COLOR_YELLOW
            # Compute bounding box points
            bb = w.bounding_box.extent
            scale_pedestrians_extent = self.params.env.sensors.scale_pedestrians_extent
            corners = [
                scale_pedestrians_extent * carla.Location(x=-bb.x, y=-bb.y),
                scale_pedestrians_extent * carla.Location(x=bb.x, y=-bb.y),
                scale_pedestrians_extent * carla.Location(x=bb.x + 0.2, y=0),
                scale_pedestrians_extent * carla.Location(x=bb.x, y=bb.y),
                scale_pedestrians_extent * carla.Location(x=-bb.x, y=bb.y)
            ]
            w.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p, offset=offset) for p in corners]
            surface = cv2.fillPoly(surface, np.int32([corners]), (color.r, color.g, color.b))
        return surface

    def _render_vehicles(self, surface, list_v, offset, surface_vel=None, ego=None):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = COLOR_BLUE
            if int(v.attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v.attributes['role_name'] == 'hero':
                color = COLOR_WHITE
            # Compute bounding box points
            bb = v.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x - 0.8, y=-bb.y),
                carla.Location(x=bb.x, y=0),
                carla.Location(x=bb.x - 0.8, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=-bb.y)
            ]
            v.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p, offset=offset) for p in corners]
            surface = cv2.fillPoly(surface, np.int32([corners]), (color.r, color.g, color.b))
        return surface


class MultiBirdsEyeViewRenderer(BirdsEyeViewRenderer):
    """Class that contains all the information of the carla world (in the form of pygame surfaces)"""

    def __init__(self, params, world):
        super().__init__(params, world)
        # channel_0: map
        # channel_1: waypoints
        # channel_2: traffic_lights
        # channel_3: ego
        # channel_4: vehicles
        # channel_5: walkers
        self.surface = np.zeros((self.size, self.size, 6), dtype=np.float32)
        self.surface_size = (np.shape(self.surface)[0], np.shape(self.surface)[1])
        self.rotate_interpolate = cv2.INTER_NEAREST

    def _render_map(self, surface, offset):
        """Renders the map"""
        c_idx = 0
        surface[:, :, c_idx] = self.map_surface[
                               int(offset[1]): int(offset[1] + self.size),
                               int(offset[0]): int(offset[0]) + self.size, :].mean(axis=2) / 255.0
        return surface

    def _render_waypoints(self, surface, waypoints, offset):
        """Renders the waypoints
        channel_0: map
        channel_1: waypoints
        channel_2: traffic_lights
        channel_3: ego
        channel_4: vehicles
        channel_5: walkers
        Args:
        - surface (np.array): surface to draw the waypoints
        - waypoints (list): list of waypoints to draw
        - offset (np.array): offset to draw the waypoints
        """

        c_idx = 1
        surf_ = np.zeros((self.size, self.size, 1), dtype=np.float32)  # reset channel

        corners = [self.map_image.world_to_pixel(carla.Location(x=p[0], y=p[1]), offset=offset) for p in waypoints]
        surf_ = cv2.polylines(surf_, np.int32([corners]), False, 1.0, 2)
        surface[:, :, c_idx] = surf_[:, :, 0]
        return surface

    def _render_traffic_lights(self, surface, traffic_lights, offset):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled
        channel_0: map
        channel_1: waypoints
        channel_2: traffic_lights
        channel_3: ego
        channel_4: vehicles
        channel_5: walkers
        Args:
        - surface (np.array): surface to draw the waypoints
        - traffic_lights (list): list of traffic lights to draw
        - offset (np.array): offset to draw the waypoints
        """
        c_idx = 2
        surf_ = np.zeros((self.size, self.size, 1), dtype=np.float32)  # reset channel

        for tl in traffic_lights:
            pos = self.map_image.world_to_pixel(tl.get_location(), offset=offset)
            radius = self.map_image.world_to_pixel_width(2)  # 1.4)
            if tl.state == carla.TrafficLightState.Red:
                color = 5.0
            elif tl.state == carla.TrafficLightState.Yellow:
                color = 4.0
            elif tl.state == carla.TrafficLightState.Green:
                color = 3.0
            elif tl.state == carla.TrafficLightState.Off:
                color = 2.0
            else:
                color = 1.0
            surf_ = cv2.circle(surf_, (pos[0], pos[1]), radius, color / 5.0, cv2.FILLED)
        surface[:, :, c_idx] = surf_[:, :, 0]
        return surface

    def _render_vehicles(self, surface, list_v, offset, surface_vel=None, ego=None):
        """Renders the vehicles' bounding boxes"""
        c_idx_hero = 3
        c_idx = 4
        surf_ = np.zeros((self.size, self.size, 1), dtype=np.float32)  # reset channel
        surf_hero_ = np.zeros((self.size, self.size, 1), dtype=np.float32)  # reset channel
        for v in list_v:
            bb = v.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x - 0.8, y=-bb.y),
                carla.Location(x=bb.x, y=0),
                carla.Location(x=bb.x - 0.8, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=-bb.y)
            ]
            v.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p, offset=offset) for p in corners]
            if v.attributes['role_name'] == 'hero':
                surf_hero_ = cv2.fillPoly(surf_hero_, np.int32([corners]), 1.0)
            else:
                surf_ = cv2.fillPoly(surf_, np.int32([corners]), 1.0)
        surface[:, :, c_idx_hero] = surf_hero_[:, :, 0]
        surface[:, :, c_idx] = surf_[:, :, 0]
        return surface

    def _render_walkers(self, surface: np.array, list_w, offset, surface_vel=None):
        """Renders the walkers' bounding boxes"""
        c_idx = 5
        surf_ = np.zeros((self.size, self.size, 1), dtype=np.float32)  # reset channel

        for w in list_w:
            # Compute bounding box points
            bb = w.bounding_box.extent
            scale_pedestrians_extent = self.params.env.sensors.scale_pedestrians_extent
            corners = [
                scale_pedestrians_extent * carla.Location(x=-bb.x, y=-bb.y),
                scale_pedestrians_extent * carla.Location(x=bb.x, y=-bb.y),
                scale_pedestrians_extent * carla.Location(x=bb.x + 0.2, y=0),
                scale_pedestrians_extent * carla.Location(x=bb.x, y=bb.y),
                scale_pedestrians_extent * carla.Location(x=-bb.x, y=bb.y)
            ]
            w.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p, offset=offset) for p in corners]
            surf_ = cv2.fillPoly(surf_, np.int32([corners]), 1.0)
        surface[:, :, c_idx] = surf_[:, :, 0]
        return surface
