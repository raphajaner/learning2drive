import copy
import time
from collections import defaultdict
from envs.carla_gym.misc import get_lane_dis
import numpy as np
from numpy import random
import psutil, os, signal, subprocess
import logging
import carla
import nvsmi

from envs.carla_gym.sensors.sensor_manager import SensorManager
from envs.carla_gym.actors.actor_manager import ActorManager
from envs.carla_gym.misc import is_used, get_free_docker_ports


class CarlaManager:
    def __init__(self, params, verbose=1, seed=1, alloc=None):
        """ Manages the connection between a Carla server and corresponding client."""
        self.params = copy.deepcopy(params)
        self.alloc = alloc
        self.server_port = None
        self.tm_port = None
        self.seed = seed + 1
        self.current_w_frame = None
        self.synchronous_mode = None
        self.dt = params.env.sim.dt

        # Do the setup by starting the server and connecting to it
        self.server_process = self.start_server()
        self.client = self.connect_client()
        assert self.client is not None

        # Get the world object
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        if self.params.viz.carla_no_rendering:
            self.set_no_rendering_mode()
        self.spectator = self.world.get_spectator()
        self.set_world_params()

        # Managers to efficiently create and close actors/sensors
        self.actor_manager = ActorManager(self.params, self.client)
        self.sensor_manager = SensorManager(self.params, self.client)
        self.traffic_manager = None
        self.tm_running = False

        self.set_synchronous_mode(self.params)

    def start_server(self):
        # Taken from https://github.com/carla-simulator/rllib-integration
        """Start a server on a random port"""

        if self.params.async_env and self.params.docker:
            time.sleep(self.params.setup.docker_carla_wait * self.seed)
            # async env starts initial env and then expands, which leads to already blocked ports
            self.alloc[0:3] = get_free_docker_ports(base_port=self.params.setup.docker_base_port + 2 * self.seed,
                                                    num_ports=3)
            print(f"Mode: Async Env -> Reallocating ports: {self.seed}, {self.alloc[0:3]}")

        if self.params.docker:
            self.server_port = self.alloc[0]
            server_command = ["docker run",
                              "--rm",
                              "--gpus \'\"device={}\"\'".format(self.alloc[3]),
                              "--user carla",
                              "-p {}:{} -p {}:{} -p {}:{}".format(self.alloc[0],
                                                                  self.alloc[0],
                                                                  self.alloc[1],
                                                                  self.alloc[1],
                                                                  self.alloc[2],
                                                                  self.alloc[2], ),
                              # "--net=host",
                              "-v /tmp/.X11-unix:/tmp/.X11-unix:rw",
                              "carlasim/carla:0.9.14 /bin/bash",
                              "./CarlaUE4.sh -RenderOffScreen -nosound",
                              "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={}".format(
                                  self.params.setup.docker_gpu),
                              "-carla-rpc-port={}".format(self.server_port),
                              # "-quality-level={}".format(self.params.viz.quality_level),
                              ]
            server_command_text = " ".join(map(str, server_command))
        else:
            self.server_port = int(random.randint(32768, 61000 - 32) + self.seed)  # Docker goes from 32768 to 61000
            # Processes tend to start simultaneously. Use random delays to avoid problems
            time.sleep(0.1 * self.seed)

            server_port_used = is_used(self.server_port)
            stream_port_used = is_used(self.server_port + 1)
            while server_port_used and stream_port_used:
                if not server_port_used:
                    logging.debug(f"Server is using the server port: {self.server_port}.")
                if not stream_port_used:
                    logging.debug(f"Server is using the streaming port: {self.server_port + 1}")
                self.server_port += 2
                server_port_used = is_used(self.server_port)
                stream_port_used = is_used(self.server_port + 1)

            server_command = [
                f"{os.environ['CARLA_ROOT']}/CarlaUE4.sh",
                "-windowed",
                f"-ResX={800}",
                f"-ResY={800}",
                "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={}".format(0),
                f"-carla-rpc-port={self.server_port}",  # carla-world-port=N   .. self.server_port
                # f'-carla-primary-port={self.server_port + 2}'
                "-quality-level={}".format(self.params.viz.quality_level)
            ]
            if self.params.viz.render_off_screen:
                server_command += ['-RenderOffScreen']

            server_command_text = " ".join(map(str, server_command))
            print(server_command_text)  # keep this print

        logging.info(f"Starting server with bash cmd: {server_command_text}")
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )
        assert server_process.poll() is None

        if self.params.docker:

            def get_full_pname(pid):
                p = subprocess.Popen(["ps -o cmd= {}".format(pid)], stdout=subprocess.PIPE, shell=True)
                return str(p.communicate()[0])

            # # > NVIDIA 470.x
            # gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.alloc[3])

            desired_carla_mem = 4000  # 4275 MB

            # while loop that waits until CARLA instance reached sufficient VRAM
            curr_carla_vram = 0
            check_counter = 0
            while curr_carla_vram < desired_carla_mem:
                # == NVIDIA 470.x
                check_counter += 1
                for gpu_proc in nvsmi.get_gpu_processes():
                    if gpu_proc.gpu_id == str(self.alloc[3]) and str(self.server_port) in get_full_pname(gpu_proc.pid):
                        curr_carla_vram = gpu_proc.used_memory
                print(f"ENV #{self.seed}/{self.params.rl.num_envs} - CARLA GPU VRAM: {curr_carla_vram} MB")

                if check_counter == 50:
                    break

            print(f"CARLA instance started: GPU {self.alloc[3]}, TCP {self.server_port}")
        else:
            time.sleep(10)

        return server_process

    def stop_server(self):
        pid = self.server_process.pid
        group_start_thread = os.getpgid(pid)
        _ = [os.kill(p.pid, signal.SIGKILL)
             for p in psutil.process_iter()
             if "carla" in p.name().lower() and os.getpgid(p.pid) == group_start_thread]

    def connect_client(self):
        time.sleep(0.1 * self.seed)
        # Taken from https://github.com/carla-simulator/rllib-integration
        while True:
            for i in range(self.params.setup.retries_on_error):
                try:
                    client = carla.Client(self.params.setup.host, self.server_port)
                    client.set_timeout(self.params.setup.timeout)
                    client.load_world(self.params.env.town)
                    print(f'Carla server connected to port {self.server_port}.')
                    return client
                except Exception as e:
                    logging.info(
                        f" Waiting for server to be ready: {e}, attempt {i + 1} of {self.params.setup.retries_on_error}")
            self.stop_server()
            # Assign fresh ports outside of designated range
            self.alloc[0:3] = get_free_docker_ports(
                base_port=self.params.setup.docker_base_port + 3 * self.params.rl.num_envs + 10, num_ports=3)
            self.server_process = self.start_server()
            time.sleep(0.1 * self.seed)
        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error'.")

    def init_traffic_manager(self):

        self.tm_port = self.server_port + 3  # // 10 + self.server_port % 10
        while is_used(self.tm_port):
            logging.debug(f"Traffic manager's port {self.tm_port} is already being used. Checking the next one.")
            self.tm_port += 1
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.set_synchronous_mode(True)
        self.tm_running = True

        traffic_manager.set_hybrid_physics_mode(self.params.env.traffic_manager.hybrid_physics_mode)
        traffic_manager.set_hybrid_physics_radius(self.params.env.traffic_manager.hybrid_physics_radius)

        # get all traffic lights in the map
        # traffic_lights = self.world.get_actors().filter("*traffic_light*")
        # for tfl in traffic_lights:
        #     # tl.set_green_time(self.params.env.traffic_manager.green_time)
        #     # tl.set_yellow_time(self.params.env.traffic_manager.yellow_time)
        #     # tl.set_red_time(self.params.env.traffic_manager.red_time)
        #     tfl_extent = tfl.trigger_volume.extent
        #     tfl.trigger_volume.extent = carla.Vector3D(tfl_extent.x, tfl_extent.y + 20, tfl_extent.z)
        #     tfl_extent = tfl.trigger_volume.extent
        #     tfl.trigger_volume = carla.BoundingBox(tfl.get_transform().location, carla.Vector3D(tfl_extent.x, tfl_extent.y + 20, tfl_extent.z))
        #     print(tfl_extent.x, tfl_extent.y, tfl_extent.z)

        # self.traffic_manager.global_percentage_speed_difference(30.0)
        # # Setup for traffic manager
        # traffic_manager.set_global_distance_to_leading_vehicle(
        #     self.params.env.traffic_manager.global_distance_to_leading_vehicle)

        # # Dormant settings
        # settings = self.world.get_settings()
        # settings.actor_active_distance = 2000
        # self.world.apply_settings(settings)
        # traffic_manager.set_respawn_dormant_vehicles(True)
        # traffic_manager.set_boundaries_respawn_dormant_vehicles(50, 200)

        logging.debug(f"Traffic manager setup and connected to port {traffic_manager.get_port()}")
        return traffic_manager

    def tick(self, timeout=10, waypoints=None):

        # Before tick, get some old data
        old_ego_state = self.sensor_manager.get_ego_state()

        # Send tick to server to move one tick forward
        self.current_w_frame = self.world.tick()

        # Sensor data
        data = self.sensor_manager.get_data(self.current_w_frame, waypoints)
        run_red_light, at_red_tfl = self.sensor_manager.run_red_light()
        data['run_red_light'] = (self.current_w_frame, run_red_light)
        data['at_red_tfl'] = (self.current_w_frame, at_red_tfl)
        data['ped_collision'] = (self.current_w_frame, self.sensor_manager.collision_walker(data))
        data['car_collision'] = (data['collision'][0], data['collision'][1] is not None)
        data['collision'] = (data['collision'][0], np.logical_or(data['ped_collision'][1], data['car_collision'][1]))

        # Ego state
        new_ego_state = self.sensor_manager.get_ego_state()
        dis, w = get_lane_dis(waypoints, new_ego_state['ego_pos'][0], new_ego_state['ego_pos'][1])
        data['dis_to_wps'] = (data['collision'][0], dis)
        data['yaw_to_wps'] = (
            data['collision'][0],
            np.arcsin(
                np.cross(w, np.array([np.cos(new_ego_state['ego_yaw']), np.sin(new_ego_state['ego_yaw'])]))
            )
        )
        data['ego_state'] = (self.current_w_frame, new_ego_state)

        # Distance travelled by ego between ticks
        data['distance_travelled'] = (self.current_w_frame,
                                      np.linalg.norm(new_ego_state['ego_pos'] - old_ego_state['ego_pos']))

        # assert all(frame == self.current_w_frame for frame, _ in data.values())
        return data

    def set_world_params(self):
        if self.params.env.weather == 'ClearNoon':
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world.set_pedestrians_cross_factor(self.params.env.other_road_users.pedestrian_cross_factor)
        settings = self.world.get_settings()
        settings.actor_active_distance = 200
        self.world.apply_settings(settings)

    def set_spectator_camera_view(self, view=carla.Transform(), z_offset=15):
        # Get the location and rotation of the spectator through its transform
        # transform = self.spectator.get_transform()
        view.location.z += z_offset
        vec = view.rotation.get_forward_vector()
        view.location.x -= 2 * vec.x
        view.location.y -= 2 * vec.y
        view.rotation.pitch = -55
        self.spectator.set_transform(view)

    def set_synchronous_mode(self, params):
        # Set fixed simulation step for synchronous mode
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)
        self.synchronous_mode = True
        logging.debug('Carla simulation set to synchronous mode.')

    def set_asynchronous_mode(self):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        self.synchronous_mode = False
        logging.debug('Carla simulation set to asynchronous mode.')

    def set_no_rendering_mode(self):
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)
        self.world.tick()
        logging.debug('Carla simulation set to no rendering mode.')

    def clear_all_actors(self):
        self.sensor_manager.close_all_sensors()
        self.actor_manager.clear_all_actors()

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def close(self):
        self.stop_server()
        del self.actor_manager
        del self.sensor_manager
        # self.traffic_manager.set_synchronous_mode(False)
        # self.world.tick()
        # del self.traffic_manager
        # self.clear_all_actors()
