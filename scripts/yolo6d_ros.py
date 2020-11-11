#! /usr/bin/env python3
from __init__ import *

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')

class Yolo6D:

    def __init__(self, datacfg):
        # Parameters
        options          = read_data_cfg(datacfg)
        fx               = float(options['fx'])
        fy               = float(options['fy'])
        cx               = float(options['cx'])
        cy               = float(options['cy'])
        weightfile       = options['weightfile']
        modelcfg         = options['modelcfg']
        meshfile         = options['meshfile']
        gpus             = options['gpus']
        seed             = int(time.time())
        use_cuda         = True
        self.classes     = 1
        self.img_width   = 640
        self.img_height  = 480
        self.conf_thresh = 0.5
        # nms_thresh       = 0.4
        # match_thresh     = 0.5

        torch.manual_seed(seed)
        if use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        # Read intrinsic camera parameters
        self.internal_calibration = get_camera_intrinsic(cx, cy, fx, fy)

        # Read object model information, get 3D bounding box corners
        mesh     = MeshPly(os.path.join(path, meshfile))
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
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.rgb_sub.registerCallback(self.callback)


    def callback(self, Image):
        self.img = np.frombuffer(Image.data, dtype=np.uint8).reshape(Image.height, Image.width, -1)

        try:
            # estimate pose
            self.pose_estimator()
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted')


    def pose_estimator(self):
        # transform into Tensor
        data = Variable(self.transform(self.img)).cuda().unsqueeze(0)
        t2 = time.time()

        # Forward pass
        output = self.model(data).data
        t3 = time.time()

        # Using confidence threshold, eliminate low-confidence predictions
        self.all_boxes = get_region_boxes2(output, self.conf_thresh, self.classes)
        # all_boxes = do_detect(self.model, self.img, 0.1, 0.4)
        t4 = time.time()

        # For each image, get all the predictions
        boxesList = []
        boxes = self.all_boxes[0]
        print(len(boxes)-1, 'onigiri(s) found')
        for j in range(len(boxes)-1):

            # ignore 1st box (NOTE: not sure why its incorrect)
            box_pr = boxes[j+1]
            print(f'{Fore.GREEN}at confidence {Style.RESET_ALL}', str(round(float(box_pr[18])*100)) + '%')

            # Denormalize the corner predictions
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.img_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.img_height

            # Compute [R|t] by PnP
            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.internal_calibration, dtype='float32'))
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

            # Compute projections
            proj_corners_pr = np.transpose(compute_projection(self.corners3D, Rt_pr, self.internal_calibration))
            boxesList.append(proj_corners_pr)

        t5 = time.time()

        # Visualize Projections
        self.visualize(self.img, boxesList, drawCuboid=True)


    def visualize(self, img, boxesList, drawCuboid=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if drawCuboid:
            for corner in boxesList:
                img = cv2.line(img, tuple(corner[0]), tuple(corner[1]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[2]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[4]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[3]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[5]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[3]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[6]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[3]), tuple(corner[7]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[5]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[6]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[5]), tuple(corner[7]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[6]), tuple(corner[7]), (0,0,255), 1)
        cv2.imshow('yolo6d pose', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            # sys.exit()
            try:
                sys.exit(1)
            except SystemExit:
                os._exit(0)


if __name__ == '__main__':

    # initiate ROS node
    rospy.init_node('onigiriPose', anonymous=False)
    rospy.loginfo('starting onigiriPose node....')

    # run Yolo6D
    path = os.path.join(os.path.dirname(__file__), '../txonigiri6d')
    datacfg = os.path.join(path, 'txonigiri.data')
    Yolo6D(datacfg)

    # ros spin
    try:
        rospy.spin()
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print('Shutting down Yolo6D ROS node')
        cv2.destroyAllWindows()
