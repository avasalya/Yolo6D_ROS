#! /usr/bin/env python3
from __init__ import *

class Yolo6D:

    def __init__(self, datacfg, subRGBTopic, subDepthTopic, frameID):
        # parameters
        options          = read_data_cfg(datacfg)
        fx               = float(options['fx'])
        fy               = float(options['fy'])
        cx               = float(options['cx'])
        cy               = float(options['cy'])
        weightfile       = options['weightfile']
        modelcfg         = options['modelcfg']
        meshfile         = options['meshfile']
        gpus             = options['gpus']
        self.classes     = 1
        self.img_width   = 640
        self.img_height  = 480
        self.conf_thresh = 0.7
        self.frameID     = frameID

        # GPU settings
        seed = int(time.time())
        torch.manual_seed(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

        # read intrinsic camera parameters
        self.cam_mat = get_camera_intrinsic(cx, cy, fx, fy)

        # Read object model information, get 3D bounding box corners
        mesh = MeshPly(os.path.join(path, meshfile))
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices),1))].transpose()
        self.corners3D = get_3D_corners(vertices)

        # Specify model
        self.model = Darknet(os.path.join(path, modelcfg))

        # load trained weights
        self.model.load_weights(os.path.join(path, weightfile))

        # pass to GPU
        self.model.cuda()

        # set the module in evaluation mode
        self.model.eval()

        # apply transformation on the input images
        self.transform = transforms.Compose([transforms.ToTensor()])

        # subscribe to RGB topic
        self.rgb_sub = message_filters.Subscriber(subRGBTopic, Image)
        self.depth_sub = message_filters.Subscriber(subDepthTopic, Image)

        # self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 9)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 1)

        self.ts.registerCallback(self.callback)
        # self.rgb_sub.registerCallback(self.callback)

        # publish PoseArray, MarkerArray
        self.pose_pub = rospy.Publisher('/onigiriPose', PoseArray)
        self.marker_pub = rospy.Publisher('/onigiriPoseMarker', MarkerArray)

        # points to draw axes lines
        self.points = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, .1], [0, 0, 0]]).reshape(-1, 3)



    def callback(self, rgb, depth):
        # convert ros-msg into numpy array
        self.img = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        self.depth = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, depth.width, -1)

        # estimate pose
        try:
            self.pose_estimator()
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted{Style.RESET_ALL}')


    def pose_estimator(self):
        # transform into Tensor
        data = Variable(self.transform(self.img)).cuda().unsqueeze(0)

        # forward pass
        output = self.model(data).data

        # using confidence threshold, eliminate low-confidence predictions
        self.all_boxes = get_region_boxes2(output, self.conf_thresh, self.classes)
        # all_boxes = do_detect(self.model, self.img, 0.1, 0.4)

        boxesList = []
        objsPose  = []
        boxes = self.all_boxes[0]
        print(len(boxes)-1, f'{Fore.YELLOW}onigiri(s) found{Style.RESET_ALL}')

        # for each image, get all the predictions
        for j in range(len(boxes)-1):

            # ignore 1st box (NOTE: not sure why its incorrect)
            box_pr = boxes[j+1]
            print(f'{Fore.GREEN}at confidence {Style.RESET_ALL}', str(round(float(box_pr[18])*100)) + '%')

            # denormalize the corner predictions
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.img_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.img_height

            # compute [R|t] by PnP
            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.cam_mat, dtype='float32'))

            # transform to align with aist ur5 ef frame
            if len(sys.argv) > 1 and sys.argv[1] == 'pnp':
                Rx = rotation_matrix(m.pi/2, [-1, 0, 0], t_pr.ravel())
                Rz = rotation_matrix(m.pi/2, [0, 0, -1], t_pr.ravel())
                offR = concatenate_matrices(Rx)[:3,:3]
                R_pr = np.dot(R_pr, offR[:3, :3])

            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

            # compute projections
            proj_corners_pr = np.transpose(compute_projection(self.corners3D, Rt_pr, self.cam_mat))
            boxesList.append(proj_corners_pr)

            # draw axes
            self.draw_axis(self.img, cv2.projectPoints(self.points, cv2.Rodrigues(R_pr)[0], t_pr, self.cam_mat, None)[0])

            # convert pose to ros-msg
            poseTransform = np.concatenate((Rt_pr, np.asarray([[0, 0, 0, 1]])), axis=0)
            quat = quaternion_from_matrix(poseTransform, True) #wxyz
            pos = t_pr.reshape(1,3)
            pose = {
                    'tx':pos[0][0],
                    'ty':pos[0][1],
                    'tz':pos[0][2],
                    'qw':quat[0],
                    'qx':quat[1],
                    'qy':quat[2],
                    'qz':quat[3]}
            objsPose.append(pose)

        # visualize Projections
        self.visualize(self.img, boxesList, drawCuboid=True)

        # publish pose as ros-msg
        self.publisher(objsPose)


    def draw_axis(self, img, axisPoints):
        img = cv2.line(img, tuple(axisPoints[3].ravel()),
                            tuple(axisPoints[0].ravel()), (255,0,0), 2)
        img = cv2.line(img, tuple(axisPoints[3].ravel()),
                            tuple(axisPoints[1].ravel()), (0,255,0), 2)
        img = cv2.line(img, tuple(axisPoints[3].ravel()),
                            tuple(axisPoints[2].ravel()), (0,0,255), 2)
        cv2.circle(img, tuple(axisPoints[3].ravel()), 5, (0, 255, 255), -1)


    def visualize(self, img, boxesList, drawCuboid=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if drawCuboid:
            for corner in boxesList:
                # draw cuboid
                linewidth = 2
                frontFace = (255,255,255)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[3]), frontFace, linewidth)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[5]), frontFace, linewidth)
                img = cv2.line(img, tuple(corner[7]), tuple(corner[3]), frontFace, linewidth)
                img = cv2.line(img, tuple(corner[7]), tuple(corner[5]), frontFace, linewidth)
                color = (0,0,255)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[1]), color, linewidth)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[2]), color, linewidth)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[4]), color, linewidth)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[3]), color, linewidth)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[6]), color, linewidth)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[5]), color, linewidth)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[6]), color, linewidth)
                img = cv2.line(img, tuple(corner[6]), tuple(corner[7]), color, linewidth)
        cv2.imshow('yolo6d pose', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            os._exit(0)


    def publisher(self, objsPose):
        marker_array = MarkerArray()

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = self.frameID
        poses = objsPose

        for p in range(len(poses)):
            pose2msg = Pose()
            pose2msg.position.x = poses[p]['tx']
            pose2msg.position.y = poses[p]['ty']
            pose2msg.position.z = poses[p]['tz']
            pose2msg.orientation.w = poses[p]['qw']
            pose2msg.orientation.x = poses[p]['qx']
            pose2msg.orientation.y = poses[p]['qy']
            pose2msg.orientation.z = poses[p]['qz']
            pose_array.poses.append(pose2msg)
            print(f'{Fore.RED} poseArray{Style.RESET_ALL}', pose_array.poses[p])

            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frameID
            marker.type = marker.CUBE
            marker.lifetime = rospy.rostime.Duration(0.1)
            marker.action = marker.ADD
            marker.id = p
            marker.scale.x = 0.07
            marker.scale.y = 0.07
            marker.scale.z = 0.02
            marker.color.a = 0.4
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = poses[p]['tx']
            marker.pose.position.y = poses[p]['ty']
            marker.pose.position.z = poses[p]['tz']
            marker.pose.orientation.w = poses[p]['qw']
            marker.pose.orientation.x = poses[p]['qx']
            marker.pose.orientation.y = poses[p]['qy']
            marker.pose.orientation.z = poses[p]['qz']

            if(p > len(poses)-1):
                marker_array.markers.pop(0)
            marker_array.markers.append(marker)

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)


if __name__ == '__main__':

    # initiate ROS node
    rospy.init_node('onigiriPose', anonymous=False)
    rospy.loginfo('starting onigiriPose node....')

    # run Yolo6D
    path = os.path.join(os.path.dirname(__file__), '../txonigiri')
    datacfg = os.path.join(path, 'txonigiri.data')

    if len(sys.argv) > 1 and sys.argv[1] == 'pnp':
        Yolo6D(datacfg, '/realsenseD435/color/image_raw',
                        '/realsenseD435/aligned_depth_to_color/image_raw',
                        'calibrated_realsenseD435_color_optical_frame')
        rospy.logwarn('publishing onigiri(s) pose for Pick-n-Place experiment')
    else:
        Yolo6D(datacfg, '/camera/color/image_raw',
                        '/camera/aligned_depth_to_color/image_raw',
                        'camera_color_optical_frame')
        rospy.logwarn('publishing onigiri(s) pose, do whatever you want with it')

    # ros spin
    try:
        rospy.spin()
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            rate.sleep(0.1)
    except KeyboardInterrupt:
        print('Shutting down Yolo6D ROS node')
        cv2.destroyAllWindows()
