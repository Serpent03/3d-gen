import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

imgL = cv2.imread('imL.jpg', 0)
imgR = cv2.imread('imr.jpg', 0)

stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=23)
disparity = stereo.compute(imgL, imgR)
disparity = cv2.convertScaleAbs(disparity)
# disparity = np.array(256 * disparity / 0x0fff,
#                             dtype=np.uint8)

pcd = []
FX_DEPTH = 464.26828003
FY_DEPTH = 463.49984741
CX_DEPTH = 319.64876017
CY_DEPTH = 234.05504395

height, width = disparity.shape
for i in range(height):
    for j in range(width):
        z = disparity[i][j]
        x = (j - CX_DEPTH) * z / FX_DEPTH
        y = (i - CY_DEPTH) * z / FY_DEPTH
        pcd.append([x, y, z])

# print(imgL.shape)
# print(imgR.shape)

pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
# Visualize:
o3d.visualization.draw_geometries([pcd_o3d])

plt.imshow(disparity)
plt.show()

