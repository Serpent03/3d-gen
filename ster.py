import numpy as np
import cv2
import matplotlib.pyplot as plt

imgL = cv2.imread('imR.jpg', 0)
imgR = cv2.imread('imL.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgR, imgL)
disp = cv2.convertScaleAbs(disparity)
depth_instensity = np.array(256 * disp / 0x0fff,
                            dtype=np.uint8)

# print(imgL.shape)
# print(imgR.shape)

plt.imshow(depth_instensity)
plt.show()

