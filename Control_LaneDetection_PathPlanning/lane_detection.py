import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import minimize

import sys
import glob
import os
import time
import cv2
import torch

try:
    sys.path.append(glob.glob('../Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# import some detection utilities
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=145, spline_smoothness=10, gradient_threshold=30, distance_maxima_gradient=5):
        self.img_shape = (200, 200, 3)
        self.car_position = np.array([self.img_shape[0]/2,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
        
    def object_detection(self, front_view_image):
        front_view_image = torch.from_numpy(np.uint8(front_view_image))
        front_view_image = torch.permute(front_view_image, (2, 0, 1))
        # Step 1: Initialize model with the best available weights
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = [preprocess(front_view_image)]

        # Step 4: Use the model and visualize the prediction
        prediction = model(batch)[0]
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(front_view_image, boxes=prediction["boxes"],
                                labels=labels,
                                colors="red",
                                width=4, font_size=30)
        box = torch.permute(box, (1, 2, 0))
        return box.cpu().detach().numpy()

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)96x96x3

        output:
            bev_image 96x96x3
        '''
        def homography_ipmnorm2g(top_view_region):
            src = np.float32([[0,0], [1, 0], [0, 1], [1, 1]])
            H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
            return H_ipmnorm2g

        bev_image = np.zeros((200, 200, 3))
        H, W = 200, 200
        top_view_region = np.float32([[50, -25], [50, 25], [0, 25], [0, -25]])

        # camera intrinsic
        camera_xyz = [-5.5, 0, 2.5]
        cam_yaw, cam_pitch, cam_roll = 0.0, 8.0, 0.0

        width = front_view_image.shape[1]
        height = front_view_image.shape[0]
        fov = 64

        # camera extrinsic
        focal = width/(2.0 * np.tan(fov * np.pi/360.0))
        K = np.identity(3)
        K[0,0] = K[1,1] = focal
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0

        RT = np.array(carla.Transform(carla.Location(*camera_xyz), carla.Rotation(yaw=cam_yaw, pitch=cam_pitch, roll=cam_roll)).get_matrix())
        RT = np.concatenate([RT[:3, 0:2], np.expand_dims(RT[:3, 3], axis=1)], axis=1)

        # Transforming Intrinsics
        R = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
        RT = R @ RT

        # Projection matrix
        P = K @ RT

        H_ipmnorm2g = homography_ipmnorm2g(top_view_region)
        H_ipmnorm2im = P @ H_ipmnorm2g

        S_im_inv = np.float32([[1/np.float32(width), 0, 0],
                            [0, 1/np.float32(height), 0],
                            [0, 0, 1]])

        M_ipm2im_norm = S_im_inv @ H_ipmnorm2im

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
                bev_image[i, j] = front_view_image[y, x]
        
        return np.uint8(bev_image) 


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the imagen at the front end of the car (e.g. pixel row 68) 
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1

        '''
        def adjust_gamma(image, gamma=1.0):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
        
            # apply gamma correction using the lookup table
            return cv2.LUT(image, table)
        
        # state_image_full = state_image_full[:, :, [2, 1, 0]]
        # cv2.imshow('main img', state_image_full)

        crop_img = state_image_full[:self.cut_size, :, :]
        gray_state_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        gray_state_image = adjust_gamma(gray_state_image, 0.5)

        return gray_state_image[::-1] 


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        '''
        gradient_x= cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_8U, 0, 1, ksize=3)

        # gradient_x[gradient_x < self.gradient_threshold] = 0
        # gradient_y[gradient_y < self.gradient_threshold] = 0
        gradient_sum = np.hypot(gradient_x, gradient_y)
        gradient_sum[gradient_sum < self.gradient_threshold] = 0
        # plt.imshow(gradient_sum, cmap='gray', vmin=0, vmax=255)

        return np.uint8(gradient_sum)

    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        num_rows = gradient_sum.shape[0]
        argmaxima = []

        # Loop over each row of the gradient image
        for i in range(num_rows):
            # Find the local maxima in the current row
            maxima, _ = find_peaks(gradient_sum[i], distance=self.distance_maxima_gradient)
            # Append the arguments of local maxima to the list
            argmaxima.append(maxima)
        # Convert the list to a numpy array
        argmaxima = np.array(argmaxima)
        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 68x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=self.distance_maxima_gradient)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < self.img_shape[0]/2:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[self.img_shape[0],  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:

                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:

                # if more than two maxima then take the two lanes next to the car, regarding least square
                # Car position is at (48, 0)
                '''
                Calculate distance between each maxima with respect to vehicle position in hortizontal axis
                Sort these distances to get closest maximas. 
                Top 2 are the lanes
                '''
                A = np.argsort((argmaxima - self.car_position[0])**2)
                # if (abs(argmaxima[A[0]]-self.car_position[0]) < 15) and (abs(argmaxima[A[1]]-self.car_position[0]) < 15):
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''
        # cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("gray_state", cv2.WINDOW_NORMAL)

        # to gray
        gray_state = self.cut_gray(state_image_full)
        cv2.imwrite("final_imgs/gray_bev.png", gray_state)
        
        
        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        cv2.imwrite("final_imgs/edge_bev.png", gradient_sum)

        # cv2.imshow("edge", gradient_sum)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        if lane_found:
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)    
            i=1
            while len(maxima)>1:
                maxima = maxima[1:]
                if len(maxima[0]) == 0:
                    continue
                maximum = np.expand_dims(maxima[0], axis=1)
                maximum = np.concatenate([maximum, i*np.ones(maximum.shape)], axis=1)

                # get distances of maximums wrt lane1 points and lane2 points
                dist_lane_1 = np.linalg.norm(maximum - lane_boundary1_points[-1], axis=1)
                dist_lane_2 = np.linalg.norm(maximum - lane_boundary2_points[-1], axis=1)
                closest_point_lane_1 = np.argsort(dist_lane_1)[0]
                closest_point_lane_2 = np.argsort(dist_lane_2)[0]
                # if closest_point_lane_1 == closest_point_lane_2:
                #     closest_point_lane_2 = np.argsort(dist_lane_2)[1]


                if dist_lane_1[closest_point_lane_1] >= 5:
                    next_pt_lane1 = [lane_boundary1_points[-1][0], i]
                else:
                    next_pt_lane1 = maximum[closest_point_lane_1]

                if dist_lane_2[closest_point_lane_2] >= 5:
                    next_pt_lane2 = [lane_boundary2_points[-1][0], i]
                else:
                    next_pt_lane2 = maximum[closest_point_lane_2]
                

                lane_boundary1_points = np.concatenate([lane_boundary1_points, [next_pt_lane1]], axis=0)
                lane_boundary2_points = np.concatenate([lane_boundary2_points, [next_pt_lane2]], axis=0)
                i += 1


            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:

                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                lane_boundary1, u1 = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]], s = self.spline_smoothness)
                # lane_boundary1 = splev(u1, tck1)

                # lane_boundary 2
                lane_boundary2, u2 = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]], s = self.spline_smoothness)
                # lane_boundary2 = splev(u2, tck2)
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+self.img_shape[0]-self.cut_size, linewidth=2, color='red')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+self.img_shape[0]-self.cut_size, linewidth=2, color='red')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+self.img_shape[0]-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,199.5))
        plt.ylim((-0.5,199.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
