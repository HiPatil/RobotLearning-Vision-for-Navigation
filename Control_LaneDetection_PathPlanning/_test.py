import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


from lane_detection import LaneDetection
from lateral_control import LateralController

from waypoint_prediction import waypoint_prediction, target_speed_prediction


# # Read BEV image
bev = cv2.imread('output/006903_bev.jpg')
# cv2.namedWindow("bev", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

cv2.imshow('bev', bev)
LD_module = LaneDetection()
LatC_module = LateralController()

# bev = cv2.resize(bev, (96, 96))
print(bev.shape)

lane1, lane2 = LD_module.lane_detection(bev)
# waypoint and target_speed prediction

waypoints = waypoint_prediction(lane1, lane2, num_waypoints=6, way_type='center')
target_speed = target_speed_prediction(waypoints)

steer = LatC_module.stanley(waypoints, target_speed)


# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

LD_module.plot_state_lane(bev, 0, fig, waypoints=waypoints)

# # cv2.imshow('IMG', np.uint8(gray_img))
cv2.waitKey(0)
cv2.destroyAllWindows()