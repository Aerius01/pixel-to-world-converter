import math
import cv2
import numpy as np
class Tracker:
    def __init__(self, particles, id, start_frame,end_frame, label):
        self.particles = particles
        self.memory = []
        self.confidence = []
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.det_frame_count = 0
        self.undet_frame_count = 0

class Particle:
    def __init__(self, x_pos, y_pos, weight, frame):
        self.position = np.array([x_pos, y_pos])
        self.weight = weight
        self.frame = frame


    def move(self, distance, angle, width, height):
        self.position[0] = round(self.position[0] + math.sin(angle) * distance)
        self.position[1] = round(self.position[1] - math.cos(angle) * distance)

        self.clamp_position(width - 1, height - 1)
        return 0  # ATD_OK

    def move_of(self, x_shift, y_shift, width, height):

        self.position[0] += x_shift
        self.position[1] += y_shift

        self.clamp_position(width - 1, height - 1)

        return 0  # ATD_OK

    def get_position(self):
        return self.position

    def get_weight(self):
        return self.weight
    def get_frame(self):
        return self.frame

    def set_position(self, x_pos, y_pos, width, height):
        self.position = np.array([x_pos, y_pos])

        self.clamp_position(width, height)
        return 0  # ATD_OK
    def copy(self):
        return Particle(self.position[0], self.position[1], self.weight)
    def set_weight(self, weight):
        self.weight = weight
        return 0  # ATD_OK

    def clamp_position(self, max_x, max_y):
        if self.position[0] < 0:
            self.position[0] = 0
        elif (self.position[0] > max_x):
            self.position[0] = max_x
        if self.position[1] < 0:
            self.position[1] = 0
        elif (self.position[1] > max_y):
            self.position[1] = max_y

        return
