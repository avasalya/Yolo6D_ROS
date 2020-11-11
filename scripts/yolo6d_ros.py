""" YOLO6D ROS Wrapper """
""" TO Test
$ create conda environment 'conda create -f environment.yml'
$ conda activate yolo6d
$ python3 fileName.py
"""

from __init__ import *

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')

class Yolo6D:

    def __init__(self, datacfg, modelcfg, weightfile):
        # Parameters
        options          = read_data_cfg(datacfg)
        self.dataDir     = options['dataDir']
        meshname         = options['mesh']
        filetype         = options['rgbfileType'] #no need later
        fx               = float(options['fx'])
        fy               = float(options['fy'])
        u0               = float(options['u0'])
        v0               = float(options['v0'])
        gpus             = options['gpus']
        seed             = int(time.time())
        use_cuda         = True
        self.img_width   = 640
        self.img_height  = 480
        self.classes     = 1
        self.conf_thresh = 0.5
        # nms_thresh       = 0.4
        # match_thresh     = 0.5

        torch.manual_seed(seed)
        if use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        # Read intrinsic camera parameters
        self.internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Read object model information, get 3D bounding box corners
        mesh      = MeshPly(meshname)
        vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        self.corners3D = get_3D_corners(vertices)

        # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
        self.model = Darknet(modelcfg)
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()

        # apply transformation on the input images
        self.transform = transforms.Compose([transforms.ToTensor()])

        # TODO
        # subscribe to RGB topic
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        info_sub     = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)

        ts           = message_filters.TimeSynchronizer([self.rgb_sub, info_sub], 10)
        self.ts.registerCallback(self.callback)



        # read still images as per the test set
        with open(os.path.join(self.dataDir, 'test.txt'), 'r') as file:
            lines = file.readlines()
        imgindex = lines[2].rstrip()
        imgpath = os.path.join(self.dataDir, 'rgb', str(imgindex) + filetype)

        # read image for visualization
        self.img = cv2.imread(imgpath)
        # cv2.imshow('yolo6d', self.img), # cv2.waitKey(1)

        # read images usin PIL
        self.img_ = Image.open(imgpath).convert('RGB')
        self.img_ = self.img_.resize((self.img_width, self.img_height))
        t1 = time.time()


    def callback(self, Image, camera_info):
        rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)

        try:
            cv2.imshow('ros to cv rgb', rgb)
            cv2.waitKey(1000)
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted')


    def pose_estimator(self):
        # transform into Tensor
        self.img_ = self.transform(self.img_)
        data = Variable(self.img_).cuda().unsqueeze(0)
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
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            sys.exit()



def main():
    rospy.init_node('onigiriPose', anonymous=False)
    rospy.loginfo('starting onigiriPose node....')

    #v3.2(95.24%) < v4.1(95.87%) < v4.2(97.14%) == v4.3
    weightfile = '../txonigiri6d/modelv4.3.weights'
    datacfg    = '../txonigiri6d/txonigiri-test.data'
    modelcfg   = '../txonigiri6d/yolo-pose.cfg'

    """ run Yolo6D """
    y6d = Yolo6D(datacfg, modelcfg, weightfile)
    y6d.pose_estimator()

    try:
        rospy.spin()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print('Shutting down Yolo6D ROS node')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()