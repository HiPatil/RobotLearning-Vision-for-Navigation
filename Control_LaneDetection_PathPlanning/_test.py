import cv2
import numpy as np
import matplotlib.pyplot as plt


from lane_detection import LaneDetection

cv2.namedWindow("IMG", cv2.WINDOW_NORMAL)


# # Read BEV image
img = cv2.imread('output.jpg')
# cv2.imshow('img', np.uint8(img))

# resize image, if not in shape (96, 96)
if img.shape != (96, 96):
    print('Resizing')
    img = cv2.resize(img, (96, 96))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Crop out the car in BEV image
# img = img[:68, :]

# print(img.shape)
# cv2.imshow('IMG', np.uint8(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

LD_module = LaneDetection()

# gray_img = LD_module.cut_gray(img)
# # cv2.imshow('gray', np.uint8(gray_img))

# gradient_sum = LD_module.edge_detection(gray_img)

# something = LD_module.find_maxima_gradient_rowwise(gradient_sum)

out = LD_module.lane_detection(img)

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

LD_module.plot_state_lane(img, 0, fig)

# cv2.imshow('IMG', np.uint8(gray_img))
cv2.waitKey(0)
cv2.destroyAllWindows()