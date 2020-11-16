# Yolo6D_ROS
ROS wrapper for Singleshotpose on custom dataset

****
* tested on Ubuntu 18.04, ROS Melodic, RTX 2080-Ti, CUDA 10.1N, Python3.7, PyTorch 1.4.1
* git clone in your catkin_ws https://github.com/avasalya/Yolo6D_ROS.git
* refer `environment.yml` for other anaconda packages

## adapted from
* https://github.com/microsoft/singleshotpose

## create conda environment
* `conda env create -f environment.yml`
<!-- * install following lib manually
`open3d`,
`rospkg`,
`chainer_mask_rcn`,
`pyrealsense2` -->

## install realsense ROS package
* https://github.com/IntelRealSense/realsense-ros

## download and unzip `txonigiri` within main directory
* https://www.dropbox.com/sh/wkmqd0w1tvo4592/AADWt9j5SjiklJ5X0dpsSILAa?dl=0

## change intrinsic parameters as per your camera
* fx, fy, cx, cy
	* `txonigiri/txonigiri.data`


<br />

# RUN
### 1. launch camera
* `roslaunch realsense2_camera rs_rgbd.launch align_depth:=true`

### 2. launch rviz along with publisher/subscriber services
*  it publishes estimated pose as geometry_msgs/PoseArray
* `roslaunch yolo6d_ros yolo6d.launch`
*  also possible via
	* `roslaunch yolo6d_ros rviz.launch`
    	* `rosrun yolo6d_ros yolo6d_ros.py`
    * or simply `python3 scripts/yolo6d_ros.py`



<!-- <br />

# Known issues -->
