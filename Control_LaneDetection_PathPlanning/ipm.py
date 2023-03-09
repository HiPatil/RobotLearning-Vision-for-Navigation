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
	src = np.float32([[0,0],
					[1, 0],
					[0, 1],
					[1, 1]])
	H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
	return H_ipmnorm2g

img = cv2.imread('00005208.jpg')

# camera intrinsic
camera_xyz = [1.6, 0, 1.7]
RT = np.array(carla.Transform(carla.Location(*camera_xyz)).get_matrix())
RT = np.concatenate([RT[:3, 0:2], np.expand_dims(RT[:3, 3], axis=1)], axis=1)
print('Intrinsic \n', RT)

# camera extrinsic
fov = 90
focal = img.shape[1]/(2.0 * np.tan(fov * np.pi/360.0))
K = np.identity(3)
K[0,0] = K[1,1] = focal
K[0, 2] = img.shape[1] / 2.0
K[1, 2] = img.shape[0] / 2.0

print('\nExtrinsic: \n', K)

# Transforming Intrinsics
R = np.array([[0, 1, 0],
		      		[0, 0, -1],
					[1, 0, 0]])
RT = R @ RT

P = K @ RT
print('\nProjection\n', P)

# Output parameters
output_size = (400, 400)

# image_roi
# roi = np.float32([[0, 570], [570, 380], [680, 380], [950, 570]])
roi = np.float32([[0, 570], [570, 380], [680, 380], [950, 570]])

#bev_roi
# bev_roi = np.float32([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]])
bev_roi = np.float32([[50, -25], [50, 25], [0, 25], [0, -25]])
# bev_roi = np.float32([[0, 570], [570, 380], [680, 380], [950, 570]])

H_ipmnorm2g = homography_ipmnorm2g(bev_roi)
H_ipmnorm2im = P @ H_ipmnorm2g
print(H_ipmnorm2im)

S_im_inv = np.float32([[1/np.float32(img.shape[1]), 0, 0],
					[0, 1/np.float32(img.shape[0]), 0],
			   		[0, 0, 1]])

M_ipm2im_norm = S_im_inv @ H_ipmnorm2im
print(M_ipm2im_norm)

# Visulization
M = torch.zeros((1, 3, 3))
M[0] = torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

linear_points_W = torch.linspace(0, 1-1/output_size[1], output_size[1])
linear_points_H = torch.linspace(0, 1-1/output_size[0], output_size[0])

base_grid = torch.zeros((output_size[0], output_size[1], 3))
base_grid[:, :, 0] = torch.outer(torch.ones(output_size[0]), linear_points_W)
base_grid[:, :, 1] = torch.outer(linear_points_H, torch.ones(output_size[1]))
base_grid[:, :, 2] = 1

grid = torch.matmul(base_grid.view(output_size[0]*output_size[1], 3), M.transpose(1, 2))
lst = grid[:, :, 2:].squeeze(0).squeeze(1).numpy() >= 0
grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])

x_vals = grid[0, :, 0].numpy() * img.shape[1]
y_vals = grid[0, :, 1].numpy() * img.shape[0]

indicate_x1 = x_vals<img.shape[1]
indicate_x2 = x_vals>0

indicate_y1 = y_vals<img.shape[0]
indicate_y2 = y_vals>0

indicate = (indicate_x1 * indicate_x2 * indicate_y1 * indicate_y2 * lst)*1

output_img = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
for i in range(output_size[0]):
	for j in range(output_size[1]):
		idx = j + i*output_size[1]
		x = int(x_vals[idx])
		y = int(y_vals[idx])
		indic = indicate[idx]
		if indic == 0:
			continue
		output_img[i, j] = img[y, x]

cv2.imwrite('output.jpg', output_img)