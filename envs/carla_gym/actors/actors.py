import numpy as np


def create_ego_bp(params, world):
    ego_bp = create_vehicle_bp(world, params.env.ego.vehicle_type, color='49,8,8')
    ego_bp.set_attribute('role_name', 'hero')
    return ego_bp


def create_vehicle_bp(world, ego_vehicle_filter='vehicle.*', color=None):
    """ Create a blueprint for a vehicle
    Args:
    - world (carla.World): the world object
    - ego_vehicle_filter (str): the filter for the ego vehicle
    - color (str): the color of the vehicle
    """
    blueprints = world.get_blueprint_library().filter(ego_vehicle_filter)
    blueprint_library = [x for x in blueprints
                         if 'carlamotors' not in x.id
                         and 'ambulance' not in x.id
                         and 'fusorosa' not in x.id
                         and 'cybertruck' not in x.id
                         and int(x.get_attribute('number_of_wheels')) == 4]

    vehicle_bp = np.random.choice(blueprint_library)

    if vehicle_bp.has_attribute('color'):
        if not color:
            color = np.random.choice(vehicle_bp.get_attribute('color').recommended_values)
        vehicle_bp.set_attribute('color', color)
    vehicle_bp.set_attribute('role_name', 'autopilot')

    return vehicle_bp


def create_walker_bp(world):
    """ Create a blueprint for a walker
    Args:
    - world (carla.World): the world object
    """
    walker_bps = world.get_blueprint_library().filter('walker.*')
    walker_bps_list = [x for x in walker_bps if x.has_attribute('is_invincible')]
    walker_bp = np.random.choice(walker_bps_list)
    walker_bp.set_attribute('is_invincible', 'True')
    assert walker_bp.get_attribute('is_invincible').as_bool()
    return walker_bp


def create_walker_controller_bp(world):
    """ Create a blueprint for a walker controller
    Args:
    - world (carla.World): the world object
    """
    return world.get_blueprint_library().find('controller.ai.walker')
