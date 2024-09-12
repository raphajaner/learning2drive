import math
import numpy as np
import carla
import pygame
from matplotlib.path import Path
import skimage
import psutil
import os
import signal
import socket


def next_free_port(min_port=32768, max_port=61000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = min_port
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


def get_free_docker_ports(base_port=32768, num_ports=100):
    """Get a list of free ports"""
    ports = list()
    curr_port = base_port
    while len(ports) < num_ports:
        curr_port = next_free_port(min_port=curr_port, max_port=61000)
        ports.append(curr_port)
        curr_port += 1
    return ports


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    main_process = psutil.Process(os.getpid())
    [os.kill(p.pid, signal.SIGKILL) for p in main_process.children(recursive=True) if "carla" in p.name().lower()]
    os.getpgid(main_process.pid)


def is_used(port):
    """Checks whether a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_pos(vehicle):
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_info(vehicle):
    """
    Get the full info of a vehicle
    :param vehicle: the vehicle whose info is to get
    :return: a tuple of x, y positon, yaw angle and half length, width of the vehicle
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    info = (x, y, yaw, l, w)
    return info


def transform_actor_to_ego_coords(ego_veh, actor):
    """
    Transform actor's global coordinates to ego coordinates
    :param ego_veh: ego vehicle
    :param actor: actor whose coordinates are to transform
    :return: actor's coordinates in ego coordinate
    """
    ego_loc = ego_veh.get_location()
    ego_yaw = ego_veh.get_transform().rotation.yaw / 180 * np.pi
    actor_loc = actor.get_location()

    # Translate to ego coordindates
    x = actor_loc.x - ego_loc.x
    y = actor_loc.y - ego_loc.y
    yaw_in_ego = actor.get_transform().rotation.yaw / 180 * np.pi - ego_yaw

    # Rotate to ego coordinates
    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                  [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    xy_in_ego = R.dot(np.array([x, y]))

    # Rotate velocity to ego coordinates
    v = actor.get_velocity()
    v_in_ego = R.dot(np.array([v.x, v.y]))

    # Get angular velocity
    w = actor.get_angular_velocity()
    w_in_ego = R.dot(np.array([w.x, w.y]))

    # Rotate acceleration to ego coordinates
    a = actor.get_acceleration()
    a_in_ego = R.dot(np.array([a.x, a.y]))

    # Rotate actor bounding box to ego_actor
    local_bb = actor.bounding_box
    bb_world_coordinates = local_bb.get_world_vertices(actor.get_transform())
    bb_corners_in_ego = []
    for p_idx, p in enumerate(bb_world_coordinates):
        if p_idx in [0, 2, 4, 6]:
            p.x -= ego_loc.x
            p.y -= ego_loc.y
            bb_corners_in_ego.append(R.dot(np.array([p.x, p.y])))

    return xy_in_ego, yaw_in_ego, v_in_ego, w_in_ego, a_in_ego, np.array(bb_corners_in_ego)


def transform_planar_pose_global2ego_planar(ego_pose, other_pose):
    """
    Transform vehicle to ego coordinate
    :param other_pose: surrounding vehicle's global pose
    :param ego_pose: ego vehicle pose
    :return: tuple of the pose of the surrounding vehicle in ego coordinate
    """
    x, y, v_x, v_y, yaw = other_pose
    ego_x, ego_y, ego_v_x, ego_v_y, ego_yaw = ego_pose
    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                  [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
    yaw_local = yaw - ego_yaw

    local_pose = (vec_local[0], vec_local[1], yaw_local)
    return local_pose


def get_pixel_info(local_info, d_behind, obs_range, image_size):
    """
    Transform local vehicle info to pixel info, with ego placed at lower center of image.
    Here the ego local coordinate is left-handed, the pixel coordinate is also left-handed,
    with its origin at the left bottom.
    :param local_info: local vehicle info in ego coordinate
    :param d_behind: distance from ego to bottom of FOV
    :param obs_range: length of edge of FOV
    :param image_size: size of edge of image
    :return: tuple of pixel level info, including (x, y, yaw, l, w) all in pixels
    """
    x, y, yaw, l, w = local_info
    x_pixel = (x + d_behind) / obs_range * image_size
    y_pixel = y / obs_range * image_size + image_size / 2
    yaw_pixel = yaw
    l_pixel = l / obs_range * image_size
    w_pixel = w / obs_range * image_size
    pixel_tuple = (x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel)
    return pixel_tuple


def get_poly_from_info(info):
    """
    Get polygon for info, which is a tuple of (x, y, yaw, l, w) in a certain coordinate
    :param info: tuple of x,y position, yaw angle, and half length and width of vehicle
    :return: a numpy array of size 4x2 of the vehicle rectangle corner points position
    """
    x, y, yaw, l, w = info
    poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
    return poly


def get_pixels_inside_vehicle(pixel_info, pixel_grid):
    """
    Get pixels inside a vehicle, given its pixel level info (x, y, yaw, l, w)
    :param pixel_info: pixel level info of the vehicle
    :param pixel_grid: pixel_grid of the image, a tall numpy array pf x, y pixels
    :return: the pixels that are inside the vehicle
    """
    poly = get_poly_from_info(pixel_info)
    p = Path(poly)  # make a polygon
    grid = p.contains_points(pixel_grid)
    isinPoly = np.where(grid == True)
    pixels = np.take(pixel_grid, isinPoly, axis=0)[0]
    return pixels


# @njit
def get_lane_dis(waypoints, x, y):
    """
    Calculate distance from (x, y) to waypoints.
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :return: a tuple of the distance and the closest waypoint orientation
    """
    dis_min = 1000
    waypt = waypoints[0]
    for pt in waypoints:
        d = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
        if d < dis_min:
            dis_min = d
            waypt = pt
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])

    if lv == 0:
        dis = 0
    else:
        cross = np.cross(w, vec / lv)
        dis = - lv * cross
    if np.isnan(dis):
        print("dis is nan.")
        # write that info to a file
        with open("nan_dis.txt", "a") as f:
            f.write("waypoints: {}, x: {}, y: {}, waypt: {}, vec: {}, lv: {}, w: {}, cross: {}\n".format(
                waypoints, x, y, waypt, vec, lv, w, cross
            ))
        dis = 0
    return dis, w


def get_preview_lane_dis(waypoints, x, y, idx=2):
    """
    Calculate distance from (x, y) to a certain waypoint
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :param idx: index of the waypoint to which the distance is calculated
    :return: a tuple of the distance and the waypoint orientation
    """
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    try:
        cross = np.cross(w, vec / lv)
    except:
        raise ValueError("w: {}, vec: {}, lv: {}".format(w, vec, lv))
    dis = - lv * cross
    return dis, w


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    try:
        theta = np.dot(forward_vector, target_vector) / norm_target
        if np.abs(theta) > 1.0:
            return False
        d_angle = math.degrees(math.acos(theta))
    except:
        raise ValueError("forward_vector: {}, target_vector: {}, norm_target: {}".format(forward_vector, target_vector,
                                                                                         norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, reference_location, reference_rotation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param reference_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    # Old approach
    target_vector = np.array([target_location.x - reference_location.x, target_location.y - reference_location.y])
    norm_target = np.linalg.norm(target_vector)

    try:
        forward_vector = np.array([math.cos(math.radians(reference_rotation)),
                                   math.sin(math.radians(reference_rotation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))
    except:
        raise ValueError(
            "1 forward_vector: {}, target_vector: {}, norm_target: {}".format(forward_vector, target_vector,
                                                                              norm_target))

    return norm_target, d_angle


def compute_signed_angle(target_location, reference_transform):
    """
    Compute relative angle and distance between a target_location and a current_location
    :param target_location: location of the target object
    :param reference_transform: transform of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    reference_location = reference_transform.location
    reference_rotation = np.radians(reference_transform.rotation.yaw)

    # Vector from the current location to the target location
    vec_target = target_location.__sub__(reference_location)
    vec_target.z = 0
    vec_target_norm = vec_target.make_unit_vector()

    # 1. Opt for calculation:
    target_rotation = np.arctan2(vec_target.y, vec_target.x)
    angle_rel_verify = np.degrees(normalize_angle(target_rotation - reference_rotation))

    # 2. Opt for calculation
    for_vec_reference = reference_transform.get_forward_vector()
    for_vec_reference.z = 0
    for_vec_reference_norm = for_vec_reference.make_unit_vector()

    # Required since result is sometimes larger than 1 due to wrong numerical problems which would case the arccos to
    # return a NaN
    length_projection = np.clip(for_vec_reference_norm.dot(vec_target_norm), -1, 1)
    angle_rel = np.degrees(np.arccos(length_projection))

    # Issue: arccos is always positive, no info about vec_ego_to_actor orientation
    # Use cross product to obtain info about orientation, if z component points down (i.e., is negative) then
    # the actor is left of the ego, i.e., the angle has to be negative in Carla's left-hand system
    if for_vec_reference_norm.cross(vec_target_norm).z < 0:
        angle_rel *= -1

    assert abs(angle_rel - angle_rel_verify) < 1.1, "Signed angle calculation is broken!"

    return angle_rel


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def distance_vehicle_no_transform_wp(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint[0] - loc.x
    dy = waypoint[1] - loc.y

    return math.sqrt(dx * dx + dy * dy)


def set_carla_transform(pose):
    """
    Get a carla transform object given pose.
    :param pose: list if size 3, indicating the wanted [x, y, yaw] of the transform
    :return: a carla transform object
    """
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform


def display_to_rgb(display, obs_size):
    """
    Transform image grabbed from pygame display to an rgb image uint8 matrix
    :param display: pygame display input
    :param obs_size: rgb image size
    :return: rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
    rgb = skimage.transform.resize(rgb, (obs_size, obs_size))  # resize
    rgb = rgb * 255
    return rgb


def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    if rgb.shape[0] != display_size:
        display = skimage.transform.resize(rgb, (display_size, display_size))
    else:
        display = rgb.astype(np.float64)
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def get_max_steering_angle_carla_vehicle(self, carla_vehicle):
    # For each Wheel Physics Control, print maximum steer angle
    physics_control = carla_vehicle.get_physics_control()
    wheels = [w for w in physics_control.wheels]
    for wheel in wheels:
        print(wheel.max_steer_angle)
    return max(wheels)


def categ2onehot(category2int, category_str):
    main_category = category_str.split('.')[0]
    if main_category in category2int.keys():
        onehot = np.zeros(len(category2int))
        onehot[category2int[main_category]] = 1
    else:
        # Get maximum value in dictionary
        incr_val = max(category2int.values()) + 1
        category2int[main_category] = incr_val
        onehot = np.zeros(len(category2int))
        onehot[incr_val] = 1

    return onehot


def get_actor_feat(actor, ego_veh, category2int):
    curr_actor_param = dict()
    # Retrieve actor in ego-vehicle coordinates (planar only, z upwards)
    actor_xy_ego, actor_yaw_ego, actor_vel_ego, actor_ang_vel_ego, actor_acc_ego, actor_bb_ego = transform_actor_to_ego_coords(
        ego_veh, actor)

    # parametrization: x, y, yaw, vx, vy, ax, ay, onehot-category (2-dim)
    curr_actor_feat_ego = np.array([actor_xy_ego[0],
                                    actor_xy_ego[1],
                                    actor_yaw_ego,
                                    actor_vel_ego[0],
                                    actor_vel_ego[1],
                                    actor_ang_vel_ego[0],
                                    actor_ang_vel_ego[1],
                                    actor_acc_ego[0],
                                    actor_acc_ego[1],
                                    ], dtype=np.float32)
    curr_actor_feat_ego = np.append(curr_actor_feat_ego, actor_bb_ego.reshape(-1, 1))
    curr_actor_feat_ego = np.append(curr_actor_feat_ego, categ2onehot(category2int, actor.type_id))
    return curr_actor_feat_ego


def filter_landmarks(path):
    # Get first waypoint from path
    first_waypoint = path[0]
    return first_waypoint


def features_to_vec(features):
    return np.array(
        [features['location'].x,
         features['location'].y,
         features['rotation'],
         features['acceleration'].x,
         features['acceleration'].y]
    )


def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """

    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


def has_passed_tfl(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Checks if the target object has passed the reference object.
    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y,
        0
    ])
    # norm_target = np.linalg.norm(target_vector)
    #
    # # If the vector is too short, we can simply stop here
    # if norm_target < 0.001:
    #     return False
    #
    # # Further than the max distance
    # if norm_target > max_distance:
    #     return False

    # min_angle = angle_interval[0]
    # max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y, 0])

    # compute cross product to get the angle sign
    cross = np.cross(forward_vector, target_vector)

    if cross[2] < 0:
        return False
    else:
        return True

    # angle = cross * math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))
    #
    # return min_angle < angle < max_angle


def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    try:
        angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))
    except:
        raise ValueError("Error in angle computation")

    return min_angle < angle < max_angle


def calc_longitudinal_distance(x1, y1, phi, x2, y2):
    # cos_phi = np.cos(phi)
    # sin_phi = np.sin(phi)
    # return (x2 - x1) * cos_phi + (z2 - z1) * sin_phi
    orientation_vector = np.array([np.cos(phi), np.sin(phi)])
    displacement_vector = np.array([x2 - x1, y2 - y1])
    longitudinal_distance = np.dot(displacement_vector, orientation_vector)
    return longitudinal_distance


def calc_lateral_distance(x1, y1, phi, x2, y2):
    orientation_vector = np.array([np.cos(phi), np.sin(phi)])
    perpendicular_vector = np.array([-orientation_vector[1], orientation_vector[0]])
    displacement_vector = np.array([x2 - x1, y2 - y1])
    lateral_distance = np.dot(displacement_vector, perpendicular_vector)
    return lateral_distance
