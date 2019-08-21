import numpy as np


class BaseState(object):

    def __init__(self):
        self.image = np.zeros((84, 84, 3), dtype=np.float32)
        self.elapsed_time = 0
        self.position = [0, 0]
        self.velocity = 0
        self.target_velocity = 0
        self.acceleration = 0
        self.distance_to_center_line = 0
        self.delta_heading = 0
        self.lane_invasion = 0
        self.lane_type = "NONE"
        self.lane_change = "NONE"
        self.opposite_lane = False
        self.junction = False
        self.collision = False
        self.terminal = False

    def __repr__(self):
        return ("Image: %s\n" +
                "Elapsed time: %f\n" +
                "Position: %s\n" +
                "Velocity: %s\n" +
                "Target velocity: %s\n" +
                "Acceleration: %s\n" +
                "Distance to center line: %s\n" +
                "Delta Heading: %d\n" +
                "Lane type: %s\n" +
                "Lane change: %s\n" +
                "Lane Invasion: %s\n" +
                "Opposite lane: %s\n" +
                "Junction: %s\n" +
                "Collision: %s\n" +
                "Terminal: %s") % (self.image.shape,
                                   self.elapsed_time,
                                   self.position,
                                   self.velocity,
                                   self.target_velocity,
                                   self.acceleration,
                                   self.distance_to_center_line,
                                   self.delta_heading,
                                   self.lane_type,
                                   self.lane_change,
                                   self.lane_invasion,
                                   self.opposite_lane,
                                   self.junction,
                                   self.collision,
                                   self.terminal)
