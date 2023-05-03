import numpy as np
import cv2
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(0)

frameHistory = []

while True:
    ret, frame = capture.read()
    frameHistory.append(frame)
    frameHistory = frameHistory[-2:]

    frameCrop = int(frame.shape[1]*0.85)
    cLeft = frame[0:-1, 0:0+frameCrop]
    cRight = frame[0:-1, frame.shape[1]-frameCrop-1:-1]

    # cLeft = frame
    # cRight = frameHistory[0]


    cLeft_8UC1 = cv2.cvtColor(cLeft, cv2.COLOR_RGB2GRAY)
    cRight_8UC1 = cv2.cvtColor(cRight, cv2.COLOR_RGB2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    np_horizontal = np.hstack((cLeft_8UC1, cRight_8UC1))
    disparity = stereo.compute(cLeft_8UC1, cRight_8UC1)

    disp = cv2.convertScaleAbs(disparity)
    depth_instensity = np.array(256 * disp / 0x0fff,
                            dtype=np.uint8)

    # disp = cv2.normalize(cv2.StereoBM.compute(cLeft_8UC1, cRight_8UC1), alpha=0, beta=255, \
    # norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # print(len(frameHistory))

    # cv2.imwrite("imR.jpg", cRight_8UC1)
    # cv2.imwrite("imL.jpg", cLeft_8UC1)

    # plt.imshow(depth_instensity)
    cv2.imshow('Image', disp)
    # plt.show()
    cv2.imshow('Video', np_horizontal)

    if cv2.waitKey(1) == 27:
        exit(0)