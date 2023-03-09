import sys
import glob
import os
import cv2
import numpy as np
import torch

try:
	sys.path.append(glob.glob('../Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla

def homography_ipmnorm2g(top_view_region):
	src = np.float32([[0,0], [1, 0], [0, 1], [1, 1]])
	H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
	return H_ipmnorm2g

bev = np.zeros((400, 400, 3))
H, W = 400, 400
top_view_region = np.float32([[50, -25], [50, 25], [0, 25], [0, -25]])

img = cv2.imread('00027126.jpg')
print(img.shape)
# camera intrinsic
camera_xyz = [-5.5, 0, 2.5]
cam_yaw, cam_pitch, cam_roll = 0.0, 8.0, 0.0

width = 480
height = 270
fov = 64

# camera extrinsic
focal = width/(2.0 * np.tan(fov * np.pi/360.0))
K = np.identity(3)
K[0,0] = K[1,1] = focal
K[0, 2] = width / 2.0
K[1, 2] = height / 2.0
print('\nExtrinsic: \n', K)

RT = np.array(carla.Transform(carla.Location(*camera_xyz), carla.Rotation(yaw=cam_yaw, pitch=cam_pitch, roll=cam_roll)).get_matrix())
RT = np.concatenate([RT[:3, 0:2], np.expand_dims(RT[:3, 3], axis=1)], axis=1)
print('Intrinsic \n', RT)

# Transforming Intrinsics
R = np.array([[0, 1, 0],
		    [0, 0, -1],
			[1, 0, 0]])
RT = R @ RT

P = K @ RT
print('\nProjection\n', P)

H_ipmnorm2g = homography_ipmnorm2g(top_view_region)
H_ipmnorm2im = P @ H_ipmnorm2g
print(H_ipmnorm2im)

S_im_inv = np.float32([[1/np.float32(width), 0, 0],
					[0, 1/np.float32(height), 0],
			   		[0, 0, 1]])

M_ipm2im_norm = S_im_inv @ H_ipmnorm2im
print(M_ipm2im_norm)

# Visulization
M = torch.zeros((1, 3, 3))
M[0] = torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

linear_points_W = torch.linspace(0, 1-1/W, W)
linear_points_H = torch.linspace(0, 1-1/H, H)

base_grid = torch.zeros((H, W, 3))
base_grid[:, :, 0] = torch.ger(torch.ones(H), linear_points_W)
base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(W))
base_grid[:, :, 2] = 1

grid = torch.matmul(base_grid.view(H*W, 3), M.transpose(1, 2))
lst = grid[:, :, 2:].squeeze(0).squeeze(1).numpy() >= 0
grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])

x_vals = grid[0, :, 0].numpy() * width
y_vals = grid[0, :, 1].numpy() * height

indicate_x1 = x_vals < width
indicate_x2 = x_vals > 0

indicate_y1 = y_vals < height
indicate_y2 = y_vals > 0

indicate = (indicate_x1 * indicate_x2 * indicate_y1 * indicate_y2 * lst)*1

for i in range(H):
	for j in range(W):
		idx = j + i*W
		x = int(x_vals[idx])
		y = int(y_vals[idx])
		indic = indicate[idx]
		if indic == 0:
			continue
		bev[i, j] = img[y, x]

cv2.imwrite('output.jpg', np.uint8(bev))