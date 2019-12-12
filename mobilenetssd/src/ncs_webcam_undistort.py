#! /usr/bin/env python

import rospy
from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import time

rospy.init_node("webcam")
rate = rospy.Rate(2) # We create a rate object of 2Hz


# Output of camera calibration
DIM=(1280, 720)
K=np.array([[665.3662159047807, 0.0, 664.5614952540346], [0.0, 668.5649720960637, 447.6043455287882], [0.0, 0.0, 1.0]])
D=np.array([[0.03272891448765657], [-0.14374793303732633], [0.11118024877954705], [0.0031267371240490836]])



while not rospy.is_shutdown():
    GRAPH = '/home/pi/catkin_ws/src/mobilenetssd/graph/graph'
    #IMAGE = '/home/samudra/catkin_ws/src/mobilenetssd/images/cat.jpg'
    CLASSES = ('background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    input_size = (300, 300)
    np.random.seed(3)
    colors = 255 * np.random.rand(len(CLASSES), 3)

    # discover our device
    devices = mvnc.EnumerateDevices()
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    # load graph onto the device
    with open(GRAPH, 'rb') as f:
        graph_file = f.read()

    graph = device.AllocateGraph(graph_file)

	# image pre-processing
    def preprocess(src):
        img = cv2.resize(src, input_size)
        img = img - 127.5
        img = img / 127.5
        return img.astype(np.float16)

    # Image undistortion
    def undistort(img, balance=0.0, dim2=None, dim3=None):
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    	assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    	if not dim2:
        	dim2 = dim1
    	if not dim3:
        	dim3 = dim1
    	scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    	scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    	#This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    	new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    	map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    	return undistorted_img

    # graph => load the image to it, return a prediction
    capture = cv2.VideoCapture(0)
    _, image = capture.read()
    height, width = image.shape[:2]

    while True:
        #image_pro = preprocess(image)
        #graph.LoadTensor(image_pro, None)
		#Commented above 2 lines of code and below code is used for undistorted view of the picamera
        stime = time.time()
        _, image = capture.read()
        image_undistort = undistort(image)
        image_pro = preprocess(image_undistort)
        graph.LoadTensor(image_pro,None)

        output, _ = graph.GetResult()

        valid_boxes = int(output[0])

        for i in range(7, 7 * (1 + valid_boxes), 7):
            if not np.isfinite(sum(output[i + 1: i + 7])):
                continue
            clss = CLASSES[int(output[i + 1])]
            conf = output[i + 2]
            color = colors[int(output[i + 1])]

            x1 = max(0, int(output[i + 3] * width))
            y1 = max(0, int(output[i + 4] * height))
            x2 = min(width, int(output[i + 5] * width))
            y2 = min(height, int(output[i + 6] * height))

            label = '{}: {:.0f}%'.format(clss, conf * 100)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            y = y1 - 5 if y1 - 15 > 15 else y1 + 18
            image = cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color, 2)
        cv2.imshow('frame', image)
        print('FPS = {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    device.CloseDevice()
    rate.sleep()
