#!/usr/bin/env python

## Import libraries
import rospy
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import sys
from tds_tensor.msg import traffic_sign

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution
IM_HEIGHT = 480   #slightly faster framerate


# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

rospy.init_node("traffic_sign_detection")
pub = rospy.Publisher('/object_detect_traffic',traffic_sign,queue_size=10 )

info = traffic_sign()
rate = rospy.Rate(1)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'traffic'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join('/home/pi/catkin_ws/src/tds_tensor/src/object_detection/traffic','frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('/home/pi/catkin_ws/src/tds_tensor/src/object_detection/traffic','label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 78

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX



# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
#Begin Count for calculating FPS
    t1 = cv2.getTickCount()
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

#Publishes the Class_Id, Class_Name, Confidence_Percent, Image_Bounding_Box 
    for index,value in enumerate(classes[0]):
         if scores[0,index] > 0.5:
             dict= category_index.get(value)
             box = np.squeeze(boxes)
             for i in range(len(boxes)):
                 ymin = (int(box[i,0]*IM_HEIGHT))
                 xmin = (int(box[i,1]*IM_WIDTH))
                 ymax = (int(box[i,2]*IM_HEIGHT))
                 xmax = (int(box[i,3]*IM_WIDTH))
                 info.id= dict['id']
                 info.label= str(dict['name'])
                 info.score= int(scores[0,index]*100)
                 info.x1=xmin
                 info.x2=xmax
                 info.y1=ymin
                 info.y2=ymax
                 pub.publish(info)
    
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    print("next inference")
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()

while not rospy.is_shutdown():

    print("ROS PREDICTION NODE RUNNING")
    rate.sleep()