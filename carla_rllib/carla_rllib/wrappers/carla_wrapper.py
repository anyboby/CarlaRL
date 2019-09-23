"""CARLA Wrapper

This script provides actor wrappers with continuous and discrete action control for the CARLA simulator.

Classes:
    * BaseWrapper - wrapper base class
    * ContinuousWrapper - actor with continuous action control
    * DiscreteWrapper - actor with discrete action control
"""
import sys
import os
import glob
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        os.environ["CARLA_ROOT"],
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc
import pygame
import queue
import numpy as np
from math import floor, ceil
import argparse
from carla_rllib.wrappers.sensors import SegmentationSensor,SegmentationSensorCustom, SegmentationSensorTags, RgbSensor, CollisionSensor, LaneInvasionSensor, RenderCamera
from carla_rllib.wrappers.states import BaseState
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import cv2

VEHICLE_MODELS = ['vehicle.audi.a2',
                  'vehicle.audi.tt',
                  'vehicle.carlamotors.carlacola',
                  'vehicle.citroen.c3',
                  'vehicle.dodge_charger.police',
                  'vehicle.jeep.wrangler_rubicon',
                  'vehicle.yamaha.yzf',
                  'vehicle.nissan.patrol',
                  'vehicle.gazelle.omafiets',
                  'vehicle.ford.mustang',
                  'vehicle.bmw.isetta',
                  'vehicle.audi.etron',
                  'vehicle.bmw.grandtourer',
                  'vehicle.mercedes-benz.coupe',
                  'vehicle.toyota.prius',
                  'vehicle.diamondback.century',
                  'vehicle.tesla.model3',
                  'vehicle.seat.leon',
                  'vehicle.lincoln.mkz2017',
                  'vehicle.kawasaki.ninja',
                  'vehicle.volkswagen.t2',
                  'vehicle.nissan.micra',
                  'vehicle.chevrolet.impala',
                  'vehicle.mini.cooperst']


CLASSES_NAMES = [
    ['Roads', 'RoadLines'],
    
    ['None', 'Buildings', 'Fences', 'Other', 'Pedestrians',
     'Poles', 'Walls', 'TrafficSigns',
     'Vegetation', 'Sidewalks'],
    
    ['Vehicles'],
]

LABELS = {
    0: 'None',
    70: 'Buildings',
    152: 'Fences',      # this one doesnt exist
    160: 'Other',
    60: 'Pedestrians',
    153: 'Poles',
    50: 'RoadLines',
    128: 'Roads',
    232: 'Sidewalks',
    35: 'Vegetation',
    142: 'Vehicles',
    156: 'Walls',
    1: 'TrafficSigns',  # this one doesnt exist
}

REVERSE_LABELS = dict(zip(LABELS.values(), LABELS.keys()))

def class_names_to_class_numbers(class_names):
    return [
        [REVERSE_LABELS[c] for c in classes]
        for classes in class_names
    ]


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


## --------------- allow dynamic memory growth to avoid cudnn init error ------------- ##
from keras.backend.tensorflow_backend import set_session #---------------------------- ##
import tensorflow as tf #------------------------------------------------------------- ##
config = tf.ConfigProto() #----------------------------------------------------------- ##
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU- ##
#config.log_device_placement = True  # to log device placement ----------------------- ##
sess = tf.Session(config=config) #---------------------------------------------------- ##
set_session(sess) # set this TensorFlow session as the default session for Keras ----- ##
## ----------------------------------------------------------------------------------- ##


class BaseWrapper(object):

    ID = 1

    def __init__(self, world, spawn_point, render=False):

        self.id = "Agent_" + str(BaseWrapper.ID)
        self._world = world
        self._map = self._world.get_map()
        self._carla_id = None
        self._vehicle = None
        self._sensors = []
        self._queues = []
        self._render_enabled = render
        self.state = BaseState()
        self._simulate_physics = True


        self._start(spawn_point, VEHICLE_MODELS[1], self.id)
        if self._render_enabled:
            self._start_render()

        BaseWrapper.ID += 1
        print(self.id + " was spawned in " + str(self._map.name) +
              " with CARLA_ID " + str(self._carla_id))

    def _start(self, spawn_point, actor_model=None, actor_name=None):
        """Spawn actor and initialize sensors"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self._world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self._world.get_blueprint_library().filter("vehicle.*"))
        if actor_name:
            blueprint.set_attribute('role_name', actor_name)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn vehicle
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._carla_id = self._vehicle.id
    
        # Set up sensors
        self._sensors.append(SegmentationSensorCustom(self._vehicle,
                                                width=800, height=800,
                                                orientation=[0, 40, -90, 0]))


        self._sensors.append(CollisionSensor(self._vehicle))
        self._sensors.append(LaneInvasionSensor(self._vehicle))

    def step(self, action):
        """Set one actor action

        Parameters:
        ----------
        action: object
            an action provided by the agent

        """
        raise NotImplementedError

    def reset(self, reset):
        """Set one actor reset

        Parameters:
        ----------
        reset: dict
            contains reset information specific to the learning goals
        """
        self.state.collisions = 0
        raise NotImplementedError

    def render(self):
        """Render the carla simulator frame"""
        self._render_camera.render(self.display)
        pygame.display.flip()

    def update_state(self, frame, start_frame, timeout):
        """Update the agent's current state

        ---Note---
        Implement your terminal conditions
        """

        # Retrieve sensor data
        self._get_sensor_data(frame, timeout)

        # Calculate non-sensor data
        self._get_non_sensor_data(frame, start_frame)

        # Check terminal conditions
        if self._is_terminal():
            self.state.terminal = True

        # Disable simulation physics if terminal
        if self.state.terminal:
            self._togglePhysics()
            return True
        else:
            return False

    def _get_sensor_data(self, frame, timeout):
        """Retrieve sensor data"""
        data = [s.retrieve_data(frame, timeout)
                for s in self._sensors]
        self.state.image = data[0]
        self.state.collision = data[1]
        if self.state.collision: self.state.collisions += 1
        self.state.lane_invasion = data[2]


    def _get_non_sensor_data(self, frame, start_frame):
        """Calculate non-sensor data"""

        # Position
        transformation = self._vehicle.get_transform()
        location = transformation.location
        self.state.position = [np.around(location.x, 2),
                               np.around(location.y, 2)]

        # Velocity
        velocity = self._vehicle.get_velocity()
        self.state.velocity = np.around(np.sqrt(velocity.x**2 +
                                                velocity.y**2 +
                                                velocity.z**2), 2)

        # Acceleration
        acceleration = self._vehicle.get_acceleration()
        self.state.acceleration = np.around(np.sqrt(acceleration.x**2 +
                                                    acceleration.y**2 +
                                                    acceleration.z**2), 2)

        # Heading wrt lane direction
        nearest_wp = self._map.get_waypoint(location,
                                            project_to_road=True)
        vehicle_heading = transformation.rotation.yaw
        wp_heading = nearest_wp.transform.rotation.yaw
        delta_heading = np.abs(vehicle_heading - wp_heading)
        if delta_heading < 180:
            self.state.delta_heading = delta_heading
        elif delta_heading > 180 and delta_heading <= 360:
            self.state.delta_heading = 360 - delta_heading
        else:
            self.state.delta_heading = delta_heading - 360

        # Opposite lane check and
        # Distance to center line of nearest (permitted) lane
        distance = np.sqrt(
            (location.x - nearest_wp.transform.location.x) ** 2 +
            (location.y - nearest_wp.transform.location.y) ** 2
        )
        if self.state.delta_heading > 90:
            self.state.opposite_lane = True
            self.state.distance_to_center_line = np.around(
                nearest_wp.lane_width - distance, 2)
        else:
            self.state.opposite_lane = False
            self.state.distance_to_center_line = np.around(distance, 2)

        # Lane type check
        wp = self._map.get_waypoint(location)
        self.state.lane_type = wp.lane_type.name

        # Lane change check
        self.state.lane_change = wp.lane_change.name

        # Junction check
        self.junction = wp.is_junction

        # Elapsed ticks
        self.state.elapsed_ticks = frame - start_frame

        # Speed limit
        speed_limit = self._vehicle.get_speed_limit()
        if speed_limit:
            self.state.speed_limit = speed_limit
        else:
            self.state.speed_limit = None

    def _is_terminal(self):
        """Check terminal conditions"""
        # Note: distance to center line depends on the town!

        if (self.state.collision or
            self.state.elapsed_ticks >= 5000):   
            return True
        else:
            return False

    def _start_render(self):
        """Start rendering camera"""
        self.display = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._render_camera = RenderCamera(self._vehicle)

    def _togglePhysics(self):
        self._simulate_physics = not self._simulate_physics
        self._vehicle.set_simulate_physics(self._simulate_physics)

    def destroy(self):
        """Destroy agent and sensors"""
        actors = [s.sensor for s in self._sensors] + [self._vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        if self._render_enabled:
            self._render_camera.sensor.destroy()


class ContinuousWrapper(BaseWrapper):
    def _start(self, spawn_point, actor_model=None, actor_name=None):
        """Spawn actor and initialize sensors"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self._world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self._world.get_blueprint_library().filter("vehicle.*"))
        if actor_name:
            blueprint.set_attribute('role_name', actor_name)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')


        

        # Spawn vehicle
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._carla_id = self._vehicle.id

        # Set up sensors
        self._sensors.append(SegmentationSensorCustom(self._vehicle,
                                                width=1000, height=1000,
                                                orientation=[0, 40, -90, 0]))


        self._sensors.append(CollisionSensor(self._vehicle))
        self._sensors.append(LaneInvasionSensor(self._vehicle))

    def step(self, action):
        """Apply steering and throttle/brake control

        action = [steer, acceleration]

        """
        control = self._vehicle.get_control()
        control.manual_gear_shift = False
        control.reverse = False
        control.hand_brake = False
        control.steer = float(action[0])

        if action[1] >= 0:
            control.brake = 0
            control.throttle = float(action[1])
        else:
            control.throttle = 0
            control.brake = -float(action[1])
        self._vehicle.apply_control(control)
        

    def reset(self, reset):
        """Reset position and controls as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation,
            steer=steer,
            acceleration=acceleration
        )

        """
        # position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)

        # controls
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        control = self._vehicle.get_control()
        control.steer = reset["steer"]
        if reset["acceleration"] >= 0:
            control.brake = 0
            control.throttle = reset["acceleration"]
        else:
            control.throttle = 0
            control.brake = -reset["acceleration"]
        self._vehicle.apply_control(control)

        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.collisions = 0
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()


class BirdsEyeWrapper(ContinuousWrapper):
    def __init__(self, world, spawn_point, render=False):
        super(BirdsEyeWrapper, self).__init__(world, spawn_point, render=render)
        model_filename = "/media/mo/Sync/Sync/Uni/Projektpraktikum Maschinelles Lernen/Workspace/ml_praktikum_ss2019_group2/semantic_birdseyeview/models/multi_model__sweep=7_decimation=2_numclasses=3_valloss=0.202.h5"
        #model_filename = "/disk/no_backup/rottach/ml_praktikum_ss2019_group2/semantic_birdseyeview/models/multi_model__sweep=10_decimation=2_numclasses=3_valloss=0.262.h5"
        self.ae = load_model(model_filename)
        self.intermediate_model = Model(inputs =self.ae.get_layer("encoder_submodel").get_input_at(0), 
                              outputs=self.ae.get_layer("encoder_submodel").get_output_at(0))

    def _start(self, spawn_point, actor_model=None, actor_name=None):
        """Spawn actor and initialize sensors"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self._world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self._world.get_blueprint_library().filter("vehicle.*"))
        if actor_name:
            blueprint.set_attribute('role_name', actor_name)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # Spawn vehicle
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._carla_id = self._vehicle.id
        IMAGE_SHAPE = (200,300)

        # Set up sensors
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
                                                orientation=[1, 3, -10, 0], id="FrontSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[0, 3, -10, -45], id="LeftSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[0, 3, -10, 45], id="RightSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[-1, 3, -10, 180], id="RearSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[0], height=IMAGE_SHAPE[1],
        #                                         orientation=[0, 40, -90, 0], id="TopSS"))                                                

        self._sensors.append(CollisionSensor(self._vehicle))
        self._sensors.append(LaneInvasionSensor(self._vehicle))

    def reset(self, reset):
        """Reset position and controls as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation,
            steer=steer,
            acceleration=acceleration
        )

        """
        # position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)
        # controls
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        control = self._vehicle.get_control()
        control.steer = reset["steer"]
        if reset["acceleration"] >= 0:
            control.brake = 0
            control.throttle = reset["acceleration"]
        else:
            control.throttle = 0
            control.brake = -reset["acceleration"]
        self._vehicle.apply_control(control)
        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.collisions = 0
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()

    def _get_sensor_data(self, frame, timeout):
        """Retrieve sensor data"""
        data = np.asarray([s.retrieve_data(frame, timeout)
                for s in self._sensors])
        trim_to_be_divisible_by = 8
        decimation = 2
        
        cam_data = data[:-2]
        dec_data = [x[::decimation, ::decimation] for x in cam_data]
        def trim(x, trim_to_be_divisible_by):
            height, width = x.shape[0:2]

            divisor = trim_to_be_divisible_by  # Easier to read
            remainder = height % divisor
            top_trim, bottom_trim = floor(remainder / 2), ceil(remainder / 2)

            remainder = width % divisor
            left_trim, right_trim = floor(remainder / 2), ceil(remainder / 2)

            return x[bottom_trim:(height-top_trim), left_trim:(width-right_trim)].astype('uint8')

        def extract_observation_for_batch(X, y, index, flip, classes_numbers):
            X_out = [x[..., index] for x in X]
            y_out = y[..., index]

            if flip:
                X_out = [np.fliplr(x) for x in X_out]
                X_out[1], X_out[2] = X_out[2], X_out[1]
                y_out = np.fliplr(y_out)

            # After mirror flipping we can transpose `y`
            y_out = np.fliplr(np.transpose(y_out, [1, 0, 2]))

            X_out = [unwrap_to_ohe(x, classes_numbers) for x in X_out]
            y_out = unwrap_to_ohe(y_out, classes_numbers)

            return X_out, y_out

        def unwrap_to_ohe(x, classes_numbers):
            x = np.stack([
                np.where(np.isin(x, classes_numbers[i]), 1, 0).astype(np.uint8)
                for i in range(len(classes_numbers))
            ])
            return np.transpose(x[..., 0], [1, 2, 0])

        trimmed_data = np.expand_dims(np.asarray(trim(dec_data[0], trim_to_be_divisible_by)), axis=3)
        #trimmed_data = np.asarray(trim(dec_data[0], trim_to_be_divisible_by))
        #trimmed_data = np.stack(trimmed_data)

        classes_numbers = class_names_to_class_numbers(CLASSES_NAMES)


        zeros = np.expand_dims(np.zeros((96,144,1), dtype="uint8"), axis=3)
        ae_input=[trimmed_data, zeros, zeros, zeros]
        #ae_input = extract_observation_for_batch(ae_input, np.zeros((96,144,1,1), dtype="uint8"), 0, True, classes_numbers)

        X = ae_input
        Y = np.zeros((144,96,1,1))

        X_final, y_final = [[] for _ in range(len(X))], []
        X_out, y_out = extract_observation_for_batch(X, Y, 0, False, classes_numbers)
        for j in range(len(X_final)):
            X_final[j].append(X_out[j])
        y_final.append(y_out)

        X_final = [np.stack(x) for x in X_final]
        y_final = np.stack(y_final)



        preds = self.ae.predict(X_final + [y_final], batch_size=1)
        encoder_layer = self.ae.get_layer(name="encoder_submodel")

        #front_ss_encoded = encoder_layer.get_output_at(0)[2:]
        front_ss_encoded = self.intermediate_model.predict(X_final[0], batch_size=1)
        #print(front_ss_encoded.shape)
        self.state.image = front_ss_encoded
        self.state.collision = data[1]
        if self.state.collision: self.state.collisions += 1
        self.state.lane_invasion = data[2]


class FrontAEWrapper(ContinuousWrapper):
    def __init__(self, world, spawn_point, render=False):
        super(FrontAEWrapper, self).__init__(world, spawn_point, render=render)
        model_filename = "/media/mo/Sync/Sync/Uni/Projektpraktikum Maschinelles Lernen/Workspace/ml_praktikum_ss2019_group2/semantic_birdseyeview/models/multi_model__sweep=7_decimation=2_numclasses=3_valloss=0.202.h5"
        self.ae = load_model(model_filename)
        # --- extract intermediate layer that contains latent space ---- #
        encoder_layer = self.ae.get_layer(name="dense_FrontRGB_1")
        encoded_in = self.ae.get_input_at(0)[0]
        encoded_out = encoder_layer.output
        self.functor = K.function([encoded_in], [encoded_out])

        # self.intermediate_model = Model(inputs =self.ae.get_layer("encoder_submodel").get_input_at(0), 
        #                       outputs=self.ae.get_layer("encoder_submodel").get_output_at(0))

    def _start(self, spawn_point, actor_model=None, actor_name=None):
        """Spawn actor and initialize sensors"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self._world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self._world.get_blueprint_library().filter("vehicle.*"))
        if actor_name:
            blueprint.set_attribute('role_name', actor_name)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # Spawn vehicle
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._carla_id = self._vehicle.id
        IMAGE_SHAPE = (200,300)

        # Set up sensors
        self._sensors.append(RgbSensor(self._vehicle,
                                                width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
                                                orientation=[1, 3, -10, 0], id="FrontRGB"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[0, 3, -10, -45], id="LeftSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[0, 3, -10, 45], id="RightSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=[1], height=IMAGE_SHAPE[0],
        #                                         orientation=[-1, 3, -10, 180], id="RearSS"))
        # self._sensors.append(SegmentationSensorTags(self._vehicle,
        #                                         width=IMAGE_SHAPE[0], height=IMAGE_SHAPE[1],
        #                                         orientation=[0, 40, -90, 0], id="TopSS"))                                                

        self._sensors.append(CollisionSensor(self._vehicle))
        self._sensors.append(LaneInvasionSensor(self._vehicle))

    def reset(self, reset):
        """Reset position and controls as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation,
            steer=steer,
            acceleration=acceleration
        )

        """
        # position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)
        # controls
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        control = self._vehicle.get_control()
        control.steer = reset["steer"]
        if reset["acceleration"] >= 0:
            control.brake = 0
            control.throttle = reset["acceleration"]
        else:
            control.throttle = 0
            control.brake = -reset["acceleration"]
        self._vehicle.apply_control(control)
        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.collisions = 0
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()

    def _get_sensor_data(self, frame, timeout):
        """Retrieve sensor data"""
        data = np.asarray([s.retrieve_data(frame, timeout)
                for s in self._sensors])
        trim_to_be_divisible_by = 8
        decimation = 2
        
        cam_data = data[:-2]
        dec_data = [x[::decimation, ::decimation] for x in cam_data]
        # debug block
        # print("decimated data: " + str(dec_data[0].shape))
        # cv2.imshow("testdec", dec_data[0])
        # cv2.waitKey(1)

        def trim(x, trim_to_be_divisible_by):
            height, width = x.shape[0:2]

            divisor = trim_to_be_divisible_by  # Easier to read
            remainder = height % divisor
            top_trim, bottom_trim = floor(remainder / 2), ceil(remainder / 2)

            remainder = width % divisor
            left_trim, right_trim = floor(remainder / 2), ceil(remainder / 2)

            return x[bottom_trim:(height-top_trim), left_trim:(width-right_trim)].astype('uint8')

        def extract_observation_for_batch(X, y, index, flip, classes_numbers):
            X_out = [x[..., index] for x in X]
            y_out = y[..., index]

            if flip:
                X_out = [np.fliplr(x) for x in X_out]
                X_out[1], X_out[2] = X_out[2], X_out[1]
                y_out = np.fliplr(y_out)

            # After mirror flipping we can transpose `y`
            y_out = np.fliplr(np.transpose(y_out, [1, 0, 2]))

            #unwrap from 1 channel SS to number of classes (only if X/Y is ss with 1 channel)
            for i in range(len(X_out)):
                X_out[i] = unwrap_to_ohe(X_out[i], classes_numbers) if X_out[i].shape[2] is 1 else X_out[i] 
            #X_out = [unwrap_to_ohe(x, classes_numbers) for x in X_out]
            y_out = unwrap_to_ohe(y_out, classes_numbers)

            return X_out, y_out

        def unwrap_to_ohe(x, classes_numbers):
            x = np.stack([
                np.where(np.isin(x, classes_numbers[i]), 1, 0).astype(np.uint8)
                for i in range(len(classes_numbers))
            ])
            return np.transpose(x[..., 0], [1, 2, 0])

        trimmed_data = np.expand_dims(np.asarray(trim(dec_data[0], trim_to_be_divisible_by)), axis=3)
        
        # debug block
        # print("trimmed_data " + str(trimmed_data[:,:,:,0].shape))
        # cv2.imshow("testtrim", trimmed_data[:,:,:,0])
        # cv2.waitKey(1)

        classes_numbers = class_names_to_class_numbers(CLASSES_NAMES)

        zeros = np.expand_dims(np.zeros((96,144,3), dtype="uint8"), axis=3)

        X = [trimmed_data, zeros, zeros, zeros]
        Y = np.zeros((144,96,3,1))

        X_final, y_final = [[] for _ in range(len(X))], []
        X_out, y_out = extract_observation_for_batch(X, Y, 0, False, classes_numbers)
        for j in range(len(X_final)):
            X_final[j].append(X_out[j])
        y_final.append(y_out)

        X_final = [np.stack(x) for x in X_final]
        y_final = np.stack(y_final)

        # debug block
        # print("X_final[0] " + str(X_final[0].shape))
        # print("X_final[0][0] " + str(X_final[0][0].shape))

        # for i in range(len(X_final)):
        #     cv2.imshow("test" + str(i), X_final[i][0])
        #     cv2.waitKey(1)

        ae_input = X_final + [y_final]
        front_ss_encoded = self.functor([ae_input[0]])[0][0]

        # --- predictions not needed for latent space extaction ----
        # 
        # preds = self.ae.predict(ae_input, batch_size=1)
        # reconstructed_ss = preds[0]
        
        # # visualize latent space vector
        # reshaped_test = front_ss_encoded.reshape(8,8,1)
        # cv2.imshow("latent", reshaped_test)
        # cv2.waitKey(1)

        # debug block
        # cv2.imshow("testpreds", front_ss_encoded[0])
        # cv2.waitKey(1)
        #print(front_ss_encoded.shape)
        
        self.state.image = front_ss_encoded
        self.state.collision = data[1]
        if self.state.collision: self.state.collisions += 1
        self.state.lane_invasion = data[2]

class DataGeneratorWrapper(ContinuousWrapper):

    def _start(self, spawn_point, actor_model=None, actor_name=None):
        super(DataGeneratorWrapper, self)._start(spawn_point)
        # Set up sensors
        self._autopilot = False
        self._sensors = []
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=200, height=300,
                                                orientation=[0, 40, -90, 0], id="TopSS"))
        self._sensors.append(RgbSensor(self._vehicle,
                                                width=300, height=200,
                                                orientation=[1, 3, -10, 0], id="FrontRGB"))
        self._sensors.append(RgbSensor(self._vehicle,
                                                width=300, height=200,
                                                orientation=[0, 3, -10, -45], id="LeftRGB"))
        self._sensors.append(RgbSensor(self._vehicle,
                                                width=300, height=200,
                                                orientation=[0, 3, -10, 45], id="RightRGB"))
        self._sensors.append(RgbSensor(self._vehicle,
                                                width=300, height=200,
                                                orientation=[-1, 3, -10, 180], id="RearRGB"))
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=300, height=200,
                                                orientation=[1, 3, -10, 0], id="FrontSS"))
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=300, height=200,
                                                orientation=[0, 3, -10, -45], id="LeftSS"))
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=300, height=200,
                                                orientation=[0, 3, -10, 45], id="RightSS"))
        self._sensors.append(SegmentationSensorTags(self._vehicle,
                                                width=300, height=200,
                                                orientation=[-1, 3, -10, 180], id="RearSS"))

        self._sensors.append(CollisionSensor(self._vehicle))

        # set autopilot for data generation
        self._vehicle.set_autopilot(self._autopilot)
        self._frames_standing = 0
    def reset(self, reset):
        """Reset position and controls as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation,
            steer=steer,
            acceleration=acceleration
        )

        """
        # position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)

        # controls
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))

        # sensors and state
        self._sensors[9].reset()
        self._frames_standing = 0
        self.state.collisions = 0
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()

    def _get_sensor_data(self, frame, timeout):
        """Retrieve sensor data"""
        self.state.storage = dict()
        cameras = (s for s in self._sensors if hasattr(s, "_id") and s._id is not "Default")
        non_cameras = (s for s in self._sensors if not hasattr(s, "_id") or s._id is "Default")
        for cam in cameras:
            self.state.storage[cam._id]=cam.retrieve_data(frame, timeout)
            self.state.image = self.state.storage[cam._id]
        if self.state.velocity < 0.1:
            self._frames_standing += 1
        data = []
        for nocam in non_cameras:
            data.append(nocam.retrieve_data(frame, timeout))
        self.state.collision = data[0]
        if self.state.collision: self.state.collisions += 1
        #self.state.lane_invasion = data[2]

    def _is_terminal(self):
        """Check terminal conditions"""
        # TODO: Adjust terminal conditions
        # @git from Moritz
        # if (self.state.collisions > 20 or
        #     self.state.distance_to_center_line > 30 or     # @MORITZ TODO maybe uncomment back to 1.8
        #         self._frames_standing > 300
        #         or self.state.elapsed_ticks >= 100000):   # @MORITZ TODO maybe uncomment back to 1000
        #     print("terminating!")
        if (self.state.collision #or
            #self.state.distance_to_center_line > 20     # @MORITZ TODO maybe uncomment back to 1.8
                or self.state.elapsed_ticks >= 5000):   # @MORITZ TODO maybe uncomment back to 1000
            return True
        else:
            return False



class DiscreteWrapper(BaseWrapper):

    def __init__(self, world, spawn_point, render=False):
        super(DiscreteWrapper, self).__init__(world, spawn_point, render)

    def step(self, action):
        """Apply discrete transformation/teleportation

        action = [x, y, rotation]

        """
        transform = carla.Transform(
            carla.Location(action[0], action[1]),
            carla.Rotation(yaw=action[2])
        )
        self._vehicle.set_transform(transform)

    def reset(self, reset):
        """Reset position as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation
        )

        """

        # Reset position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))

        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.terminal = False
        self.state.collisions = 0
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()
