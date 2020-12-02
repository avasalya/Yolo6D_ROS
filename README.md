# Yolo6D_ROS
ROS wrapper for Singleshotpose (Yolo6D) on custom dataset

****
* tested on Ubuntu 18.04, ROS Melodic, RTX 2080-Ti, CUDA 10.1, Python3.7, PyTorch 1.4.1
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
	* weights v1.xs and v2.xs were trained on synthetic dataset and v3.x onwards on real dataset

## change intrinsic parameters as per your camera
* fx, fy, cx, cy
	* `txonigiri/txonigiri.data`

## tune `conf_thresh`
* consider tuning `conf_thresh` as per your *scene* to avoid double detection and for better performance
	* change value @ [scripts/yolo6d_ros.py#L28](https://github.com/avasalya/Yolo6D_ROS/blob/6efdb5f191a70c243937c2b388e3093c6e5ebcdd/scripts/yolo6d_ros.py#L28)


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


### 3. output
* you should see output similar to this
![Alt text](img/onigiripick.png?raw=true "yolo6d pose")
![Alt text](img/yolo6dpose.png?raw=true "yolo6d pose")

<!-- <br />

# Known issues -->
