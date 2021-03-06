#! /usr/bin/env python3
from __init__ import *

class Yolo6D:

    def __init__(self, datacfg, subRGBTopic, subDepthTopic, frameID):

        # parameters
        options          = read_data_cfg(datacfg)
        self.frameID     = frameID
        self.gpus        = options['gpus']
        self.modelcfg    = options['modelcfg']
        self.meshfile    = options['meshfile']
        self.weightfile  = options['weightfile']
        self.pixel_depth = options['use_pixelD']
        self.dd_remove   = options['use_dd']
        self.NMS         = options['use_nms']
        self.pnp         = options['use_pnpG']
        self.classes     = int(options['classes'])
        self.img_width   = int(options['width'])
        self.img_height  = int(options['height'])
        self.dd_thresh   = float(options['dd'])
        self.nms_thresh  = float(options['nms'])
        self.conf_thresh = float(options['conf'])
        fx               = float(options['fx'])
        fy               = float(options['fy'])
        cx               = float(options['cx'])
        cy               = float(options['cy'])
        k1               = float(options['k1'])
        k2               = float(options['k2'])
        p1               = float(options['p1'])
        p2               = float(options['p2'])
        k3               = float(options['k3'])

        # initiate ROS node
        self.modelNname = self.weightfile.split('weights')[0] + "NMS_" + self.NMS + "_DD_" + self.dd_remove
        rospy.init_node(self.modelNname, anonymous=False)
        rospy.loginfo('starting onigiriPose node....')

        # GPU settings
        seed = int(time.time())
        torch.manual_seed(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        torch.cuda.manual_seed(seed)

        # read intrinsic camera parameters
        self.cam_mat = get_camera_intrinsic(fx, cx, fy, cy)
        self.distCoeffs = np.array([k1, k2, p1, p2, k3])

        # Read object model information, get 3D bounding box corners
        mesh = MeshPly(os.path.join(path, self.meshfile))
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices),1))].transpose()
        self.corners3D = get_3D_corners(vertices)

        # Specify model
        self.model = Darknet(os.path.join(path, self.modelcfg))

        # load trained weights
        self.model.load_weights(os.path.join(path, self.weightfile))

        # pass to GPU
        self.model.cuda()

        # set the module in evaluation mode
        self.model.eval()

        # apply transformation on the input images
        self.transform = transforms.Compose([transforms.ToTensor()])

        # subscribe to RGB topic
        self.rgb_sub = message_filters.Subscriber(subRGBTopic, Image)
        self.depth_sub = message_filters.Subscriber(subDepthTopic, Image)

        # self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 2)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)

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
        self.depth = np.frombuffer(depth.data, dtype=np.uint16).reshape(depth.height, depth.width, -1)
        #dtype=np.float32

        # visualize depth
        self.convertDepth = self.depth.copy()
        self.convertDepth = cv2.normalize(self.convertDepth, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow("depth ros2numpy", self.convertDepth), cv2.waitKey(1)

        # estimate pose
        try:
            self.pose_estimator()
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted{Style.RESET_ALL}')


    def pose_estimator(self):

        # transform image into Tensor
        data = Variable(self.transform(self.img)).cuda().unsqueeze(0)

        # forward pass
        output = self.model(data).data

        # using confidence threshold, eliminate low-confidence predictions
        self.all_boxes = get_region_boxes0(output, self.conf_thresh, self.classes)
        # print('found boxes, after removing low confidence', len(self.all_boxes))

        # apply NMS to further remove double detection
        if self.NMS == 'True':
            self.all_boxes = nms(self.all_boxes, self.nms_thresh)

        boxesList = []
        posesList = []
        sortNorms = []
        sortConfs = []
        axesList  = []
        newDepths = []

        # for each image, get all the predictions
        for j in range(0, len(self.all_boxes)):

            box_pr = self.all_boxes[j]

            # denormalize the corner predictions
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.img_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.img_height

            # compute [R|t] by PnP
            R_pr, t_pr = pnp(
                            np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),
                            corners2D_pr,
                            np.array(self.cam_mat, dtype='float32'),
                            distCoeffs = self.distCoeffs,
                            PNPGeneric=self.pnp)

            # transform to align with aist ur5 ef frame
            if len(sys.argv) > 1 and sys.argv[1] == 'pnp':
                Rx = rotation_matrix(m.pi/2, [-1, 0, 0], t_pr.ravel())
                Rz = rotation_matrix(m.pi/2, [0, 0, -1], t_pr.ravel())
                offR = concatenate_matrices(Rx)[:3,:3]
                R_pr = np.dot(R_pr, offR[:3, :3])
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

            # compute projections #NOTE: consider distCoeffs too
            # http://opencv.jp/opencv-2.1_org/py/camera_calibration_and_3d_reconstruction.html
            proj_corners_pr = compute_projection(self.corners3D, Rt_pr, self.cam_mat).transpose()
            # proj_corners_pr = cv2.projectPoints(self.corners3D, cv2.Rodrigues(R_pr)[0], t_pr, self.cam_mat, self.distCoeffs)[0]
            boxesList.append(proj_corners_pr)

            # make list of sorted conf and predicted pose(s) centeroid
            axesPoints = cv2.projectPoints(self.points, cv2.Rodrigues(R_pr)[0], t_pr, self.cam_mat, self.distCoeffs)[0]
            axesList.append(axesPoints)

            # get min/max coordinates of proj_corners
            minPt = np.min(proj_corners_pr, axis=0)
            maxPt = np.max(proj_corners_pr, axis=0)

            # gather depth of pixels within the bounding box
            # pixelDepth = self.depth[int(minPt[1]):int(minPt[0]), int(maxPt[1]):int(maxPt[0])].astype(float) #avg of bbox
            pixelDepth = self.depth[int(axesPoints[3].ravel()[1]), int(axesPoints[3].ravel()[0])].astype(float) #centroid
            # print('Depth pixels ',pixelDepth)

            # remove Zeros and NAN before taking depth mean
            nonZero = pixelDepth[pixelDepth!=0]
            # print(nonZero)

            # mean of depth pixels
            meanDepth = np.nanmean(nonZero)*.001 #mm2m
            # print('mean', meanDepth)

            # visualize just bounding boxes
            cv2.rectangle(self.img, (int(minPt[0]), int(minPt[1])),
                        (int(maxPt[0]), int(maxPt[1])), (100,243,34), 2 )
            cv2.imshow('yolo6d pose ' + self.modelNname, cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:
            #     print('stopping, keyboard interrupt')
            #     os._exit(0)

            # convert pose to ros-msg
            poseTransform = np.concatenate((Rt_pr, np.asarray([[0, 0, 0, 1]])), axis=0)
            quat = quaternion_from_matrix(poseTransform, True) #wxyz
            pos = t_pr.reshape(1,3)
            # print('before', pos)

            # take mean of both pixel and pred depths
            if self.pixel_depth == 'True':
                if not math.isnan(meanDepth):
                    # pos[0][2] = (pos[0][2] + meanDepth)/2
                    pos[0][2] = meanDepth
            # print('after pos', pos, '\n')

            pose = {
                    'tx':pos[0][0],
                    'ty':pos[0][1],
                    'tz':pos[0][2],
                    'qw':quat[0],
                    'qx':quat[1],
                    'qy':quat[2],
                    'qz':quat[3]}
            posesList.append(pose)

            # filter out low confidence double detections(dd) based on norm(xy)
            cenX = axesPoints[3].ravel()[1]/self.img_width
            cenY = axesPoints[3].ravel()[0]/self.img_height
            sortNorms.append(np.linalg.norm([cenX, cenY]))
            sortConfs.append(round(float(box_pr[18])*100))
            newDepths.append(meanDepth)

        # filter out low confidence double detections(dd)
        if self.dd_remove == 'True':
            sortNorms = sorted(sortNorms)
            sortConfs = sorted(sortConfs)
            # print('sorted norms', sortNorms)
            # print('sorted conf', sortConfs)

            indices = []
            for j in range(len(sortNorms)-1):

                # pick only double detection boxes
                # print("norm diff outside", sortNorms[j+1] - sortNorms[j])
                if (sortNorms[j+1] - sortNorms[j]) <= self.dd_thresh:

                    # print("norm diff", sortNorms[j+1] - sortNorms[j])
                    # if newDepths[j] >= newDepths[j+1]: #consider nearest
                    # remove with lower confidence
                    if (sortConfs[j+1] >= sortConfs[j]): #consider high conf
                        index = j+1
                    else:
                        index = j
                    indices.append(index)

            # remove lower conf double detection
            # print('indices', sorted(indices, reverse=True))
            for idx in sorted(indices, reverse=True):
                del axesList[idx]
                del boxesList[idx]
                del posesList[idx]

        # draw axes
        self.draw_axes(self.img, axesList)

        # visualize Projections
        self.visualize(self.img, boxesList, drawCuboid=True)

        # publish pose as ros-msg
        self.publisher(posesList)


    def draw_axes(self, img, axesList):
        for axesPoint in axesList:
            img = cv2.line(img, tuple(axesPoint[3].ravel()),
                                tuple(axesPoint[0].ravel()), (255,0,0), 2)
            img = cv2.line(img, tuple(axesPoint[3].ravel()),
                                tuple(axesPoint[1].ravel()), (0,255,0), 2)
            img = cv2.line(img, tuple(axesPoint[3].ravel()),
                                tuple(axesPoint[2].ravel()), (0,0,255), 2)
            cv2.circle(img, tuple(axesPoint[3].ravel()), 5, (0, 255, 255), -1)


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
        cv2.imshow('yolo6d pose ' + self.modelNname, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            os._exit(0)
        print(len(boxesList), f'{Fore.YELLOW}onigiri(s) found{Style.RESET_ALL}') #, end='\r')


    def publisher(self, posesList):
        marker_array = MarkerArray()

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = self.frameID
        poses = posesList

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
            # print(p, f'{Fore.RED} poseArray{Style.RESET_ALL}', pose_array.poses[p])

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
