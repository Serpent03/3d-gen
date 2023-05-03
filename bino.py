import numpy as np
import cv2
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(1)

frameHistory = []
init = 0

while True:
    ret, frame = capture.read()
    frameHistory.append(frame)
    frameHistory = frameHistory[-2:]
    init += 1

    cLeft = frame[0:-1, 0:0+500]
    cRight = frame[0:-1, frame.shape[1]-500-1:-1]

    # cLeft = frame
    # cRight = frameHistory[0]


    cLeft_8UC1 = cv2.cvtColor(cLeft, cv2.COLOR_RGB2GRAY)
    cRight_8UC1 = cv2.cvtColor(cRight, cv2.COLOR_RGB2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
    np_horizontal = np.hstack((cLeft_8UC1, cRight_8UC1))
    disparity = stereo.compute(cLeft_8UC1, cRight_8UC1)

    # print(len(frameHistory))

    # cv2.imwrite("imR.jpg", cRight_8UC1)
    # cv2.imwrite("imL.jpg", cLeft_8UC1)

    plt.imshow(disparity)
    plt.show()
    cv2.imshow('Video', np_horizontal)

    if cv2.waitKey(1) == 27:
        exit(0)