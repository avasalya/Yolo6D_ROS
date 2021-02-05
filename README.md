# Yolo6D_ROS
ROS wrapper for Singleshotpose (Yolo6D) on custom dataset

****
* tested on Ubuntu 18.04, ROS Melodic, RTX 2080-Ti, CUDA 10.1, Python3.7, PyTorch 1.4.1
* git clone in your catkin_ws https://github.com/avasalya/Yolo6D_ROS.git
* refer `environment.yml` for other anaconda packages

## adapted from
* https://github.com/microsoft/singleshotpose (original)
* https://github.com/avasalya/singleshot6Dpose (modified for my usage)

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
* https://www.dropbox.com/sh/wkmqd0w1tvo4592/AADWt9j5SjiklJ5X0dpsSILAa?dl=0 (require password)
*
	* weights v1.xs and v2.xs were trained on synthetic dataset and v3.x onwards on real dataset

## change intrinsic parameters as per your camera
* fx, fy, cx, cy
	* `txonigiri/txonigiri.data`


## consider tuning `thresholds` as per your *scene* to avoid double detection and for better performance
* change values @ [scripts/yolo6d_ros.py#L21](https://github.com/avasalya/Yolo6D_ROS/blob/a1569e1a106a3f329d20d21a6087f9b658df3fba/scripts/yolo6d_ros.py#L21) onwards
* tune `dd_thresh`
* tune `conf_thresh`
* tune `nms_thresh` (its aggressive)
* tune `softnms_thresh` (TODO)


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
![Alt text](img/yolo6dpose.png?raw=true "yolo6d pose")

<br />

* with aist-moveit pkg for Pick-n-Place
![Alt text](img/onigiripick.png?raw=true "yolo6d pose")


<br />

# Train on custom dataset
* refer this repository to train on your custom dataset https://github.com/avasalya/singleshot6Dpose

# Create your own custom dataset
* use this repository to create your own dataset for Yolo6D (develop branch) https://github.com/avasalya/RapidPoseLabels/tree/develop
 	* I have made changes in the original repository to meet the necessary requirements to produce dataset for yolo6D.
	* you can find original work here, https://github.com/rohanpsingh/RapidPoseLabels

<!-- <br />
# Known issues -->
