# Traffic Sign  Detection using Movidius Neural Compute Stick

This package has two parts tds_tensor and mobilenetssd . The package tsd_tensor can detect german traffic signs 
with a frame rate of ~0.4 FPS . The mobilenetssd can perform object detection with a frame rate of ~7 FPS

---------
## Setting up the repository
Clone this repository into a catkin workspace and run `catkin_make`. After that is
succeeded,launch the tsd_tensor.launch to perform traffic sign detection and mobilenetssd.launch
to perform object detection using the Intel Neural Compute Stick.

```bash
roslaunch tds_tensor tds_tensor.launch
```

Running the above command will launch a window where detected traffic sign from the RasPi cam os displayed 
along with the bounding box. 
It also publishes the bounding box details such as image_id, co-ordinates and class name 
onto the topic /object_detect_traffic
-----------
 
