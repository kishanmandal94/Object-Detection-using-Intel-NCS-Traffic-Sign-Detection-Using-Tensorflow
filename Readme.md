# Real Time Object Detection using Intel Movidius Neural Compute Stick and Traffic Sign Recognition using Tensorflow 

The Project is done in two parts:

Part 1: ​ The first requirement is to use the Intel Neural Compute Movidius Stick (NCS)
with RaspberryPi to learn how to utilize it as an edge computing platform and improve the
inference speed and frame rate (FPS) of the Object detector. A pre-trained Caffe model
trained on 21 classes viz. car, person, bus, etc. is deployed which uses the NCS as a
Visual Processing Unit (VPU) to give a faster inference. The object detector detects with
good accuracy with fps ~(7-8 ). Since no appropriate pre-trained object detector network
under Caffe model was available that explicitly detects traffic signs, the reference to
another deep learning framework such as ​ Tensorflow​ is done.

Part 2: ​ The focus is to detect Traffic sign using Tensorflow framework and deploying it
on the embedded platform of RaspberryPi using ROS environment. This package when
launched runs the object detector network, trained on 78 different classes of traffic signs
and can detect them at a frame rate of ~0.4 fps on Raspberry Pi3 without the NCS.

![](images/embedded_Images.png)

---------
## Setting up the repository
Clone this repository into a catkin workspace and run `catkin_make`. After that is
succeeded,launch the tsd_tensor.launch to perform traffic sign detection and mobilenetssd.launch
to perform object detection using the Intel Neural Compute Stick.

```bash
roslaunch tds_tensor tds_tensor.launch
```
---------
# Output

Running the above command will launch a window where detected traffic sign from the RasPi cam os displayed 
along with the bounding box. It also publishes the bounding box details such as image_id, co-ordinates and class name  onto the topic `object_detect_traffic`

-----------
 
