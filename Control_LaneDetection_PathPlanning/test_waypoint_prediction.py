import gym
from gym.envs.box2d.car_racing import CarRacing

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2


# # Read BEV image
img = cv2.imread('imgs/output.jpg')
# img = cv2.imread('imgs/bev_final.jpg')

cv2.imshow('img', np.uint8(img))

# resize image, if not in shape (96, 96)
if img.shape != (96, 96):
    print('Resizing')
    img = cv2.resize(img, (96, 96))

# init modules of the pipeline
LD_module = LaneDetection()

lane1, lane2 = LD_module.lane_detection(img)
waypoints = waypoint_prediction(lane1, lane2)
# print(waypoints)
speed = target_speed_prediction(waypoints)
print(speed)

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()



#     # waypoint and target_speed prediction
#     target_speed = target_speed_prediction(waypoints)

#     # reward
#     total_reward += r

#     # outputs during training
#     if steps % 2 == 0 or done:
#         print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
#         print("step {} total_reward {:+0.2f}".format(steps, total_reward))

#         LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        
#     steps += 1
#     env.render()

#     # check if stop
#     if done or restart or steps>=600: 
#         print("step {} total_reward {:+0.2f}".format(steps, total_reward))
#         break

# env.close()