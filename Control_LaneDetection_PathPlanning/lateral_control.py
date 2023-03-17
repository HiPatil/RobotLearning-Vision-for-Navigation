import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=5, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0

    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation
        waypoints = waypoints.T
        waypt_dir = waypoints[2] - waypoints[0]
        cos_psi = np.dot(waypt_dir, [0, 1])/np.linalg.norm(waypt_dir)
        psi = np.arccos(cos_psi)
        # psi = 0
        # print(psi)

        # derive cross track error as distance between desired waypoint at spline parameter equal zero to the car position
        # dt1 = (-self.gain_constant*self.previous_dt)/(np.sqrt(1+(self.gain_constant*self.previous_dt/speed)**2))
        dt1 = waypoints[1, 0] - 48.0

        # derive stanley control law
        delta_sc_t2 = psi + np.arctan(self.gain_constant*dt1/speed)

        # prevent division by zero by adding as small epsilon

        # derive damping term
        steering_angle = delta_sc_t2 - self.damping_constant * (delta_sc_t2 - self.previous_steering_angle)
        # self.previous_steering_angle = steering_angle
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






