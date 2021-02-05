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
## Label files

 Label files consist of 21 ground-truth values. We predict 9 points corresponding to the centroid and corners of the 3D object model. Additionally we predict the class in each cell. That makes 9x2+1 = 19 points. In multi-object training, during training, we assign whichever anchor box has the most similar size to the current object as the responsible one to predict the 2D coordinates for that object. To encode the size of the objects, we have additional 2 numbers for the range in x dimension and y dimension. Therefore, we have 9x2+1+2 = 21 numbers.

Respectively, 21 numbers correspond to the following: 1st number: class label, 2nd number: x0 (x-coordinate of the centroid), 3rd number: y0 (y-coordinate of the centroid), 4th number: x1 (x-coordinate of the first corner), 5th number: y1 (y-coordinate of the first corner), ..., 18th number: x8 (x-coordinate of the eighth corner), 19th number: y8 (y-coordinate of the eighth corner), 20th number: x range, 21st number: y range.

The coordinates are normalized by the image width and height: x / image_width and y / image_height. This is useful to have similar output ranges for the coordinate regression and object classification tasks.

* use this repository to create your own dataset for Yolo6D (develop branch) https://github.com/avasalya/RapidPoseLabels/tree/develop (instructions are provided)

   * I have made changes in the original repository to meet the necessary requirements to produce dataset for yolo6D.

  * please read `dataset.sh` for further instructions on how to create your own dataset

  * once you run `dataset.sh`, this will generate the `out_cur_date` folder with several contents, however to train `Yolo6D` you just need to copy `rgb`, `mask`, `label` folders, and `train.txt`, `test.txt`, `yourObject.ply` files.

  * Please refer to original work here for further support, https://github.com/rohanpsingh/RapidPoseLabels, PS: but my forked branch is not updated.

<!-- <br />
# Known issues -->
