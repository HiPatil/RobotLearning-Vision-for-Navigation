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


    def __init__(self, gain_constant=5, damping_constant=0.8):

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
        x_dir = waypoints[0, 1] - waypoints[0, 0]
        y_dir = waypoints[1, 1] - waypoints[1, 0]
        psi = np.arctan2(x_dir, y_dir)
        # print('PSI \t', psi)

        # derive cross track error as distance between desired waypoint at spline parameter equal zero to the car position
        # dt1 = (-self.gain_constant*self.previous_dt)/(np.sqrt(1+(self.gain_constant*self.previous_dt/speed)**2))
        x_t = waypoints[0, 0] - 100.0
        # print('XT \t', x_t)
        # prevent division by zero by adding as small epsilon
        eps = 1e-9
        # derive stanley control law
        delta = psi + np.arctan(self.gain_constant * x_t/ (speed+eps))
        # print('Angle \t', delta)
        # derive damping term
        damping = self.damping_constant * (delta - self.previous_steering_angle)
        
        # a =0
        # b=0
        # if delta <0 :
        #     a = delta
        #     b = -1 * delta
        # else :
        #     a = -1 * delta
        #     b = delta
        
        # damping = np.clip(damping, a - 0.1* delta, b + 0.1*delta)
        steering_angle = delta - damping
        steering_angle = np.clip(steering_angle, -0.4, 0.4) 
        # print('DAngle \t', steering_angle)
        self.previous_steering_angle = steering_angle

        return steering_angle






