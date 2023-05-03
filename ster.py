import numpy as np
import cv2
import matplotlib.pyplot as plt

imgL = cv2.imread('imR.jpg', 0)
imgR = cv2.imread('imL.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgR, imgL)

# print(imgL.shape)
# print(imgR.shape)

plt.imshow(disparity, 'gray')
plt.show()

