import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from scipy import spatial

import struct
import imghdr

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files

def calcAngularDistance(gt_rot, pr_rot):

    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff)
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e

def get_3D_corners(vertices):

    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                            np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                            cameraMatrix,
                            distCoeffs,flags=cv2.SOLVEPNP_ITERATIVE)
                            # cv2.SOLVEPNP_ITERATIVE
                            # cv2.SOLVEPNP_EPNP
                            # cv2.SOLVEPNP_UPNP
                            # cv2.SOLVEPNP_DLS

    _, R_exp, t,_ = cv2.solvePnPGeneric(points_3D,
                            np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                            cameraMatrix,
                            distCoeffs,flags=cv2.SOLVEPNP_EPNP)

    R, _ = cv2.Rodrigues(R_exp[0])
    return R, t[0]

def get_2d_bb(box, size):
    x = box[0]
    y = box[1]
    min_x = np.min(np.reshape(box, [-1,2])[:,0])
    max_x = np.max(np.reshape(box, [-1,2])[:,0])
    min_y = np.min(np.reshape(box, [-1,2])[:,1])
    max_y = np.max(np.reshape(box, [-1,2])[:,1])
    w = max_x - min_x
    h = max_y - min_y
    new_box = [x*size, y*size, w*size, h*size]
    return new_box

def compute_2d_bb(pts):
    min_x = np.min(pts[0,:])
    max_x = np.max(pts[0,:])
    min_y = np.min(pts[1,:])
    max_y = np.max(pts[1,:])
    w  = max_x - min_x
    h  = max_y - min_y
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    new_box = [cx, cy, w, h]
    return new_box

def compute_2d_bb_from_orig_pix(pts, size):
    min_x = np.min(pts[0,:]) / 640.0
    max_x = np.max(pts[0,:]) / 640.0
    min_y = np.min(pts[1,:]) / 480.0
    max_y = np.max(pts[1,:]) / 480.0
    w  = max_x - min_x
    h  = max_y - min_y
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    new_box = [cx*size, cy*size, w*size, h*size]
    return new_box

def corner_confidences(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 9 confidence values
    '''
    shape = gt_corners.size()
    nA = shape[1]
    dist = gt_corners - pr_corners
    num_el = dist.numel()
    num_keypoints = num_el//(nA*2)
    dist = dist.t().contiguous().view(nA, num_keypoints, 2)
    dist[:, :, 0] = dist[:, :, 0] * im_width
    dist[:, :, 1] = dist[:, :, 1] * im_height

    eps = 1e-5
    distthresh = torch.FloatTensor([th]).repeat(nA, num_keypoints)
    dist = torch.sqrt(torch.sum((dist)**2, dim=2)).squeeze() # nA x 9
    mask = (dist < distthresh).type(torch.FloatTensor)
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1
    conf = conf / conf0.repeat(1, num_keypoints)
    # conf = 1 - dist/distthresh
    conf = mask * conf  # nA x 9
    mean_conf = torch.mean(conf, dim=1)
    return mean_conf

def corner_confidence(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values
    '''
    dist = torch.FloatTensor(gt_corners) - pr_corners
    num_keypoints = dist.numel()//2
    dist = dist.view(num_keypoints, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height
    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=1))
    mask  = (dist < th).type(torch.FloatTensor)
    conf  = torch.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(num_keypoints, 1)
    conf  = mask * conf
    return torch.mean(conf)

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def fix_corner_order(corners2D_gt):
    corners2D_gt_corrected = np.zeros((9, 2), dtype='float32')
    corners2D_gt_corrected[0, :] = corners2D_gt[0, :]
    corners2D_gt_corrected[1, :] = corners2D_gt[1, :]
    corners2D_gt_corrected[2, :] = corners2D_gt[3, :]
    corners2D_gt_corrected[3, :] = corners2D_gt[5, :]
    corners2D_gt_corrected[4, :] = corners2D_gt[7, :]
    corners2D_gt_corrected[5, :] = corners2D_gt[2, :]
    corners2D_gt_corrected[6, :] = corners2D_gt[4, :]
    corners2D_gt_corrected[7, :] = corners2D_gt[6, :]
    corners2D_gt_corrected[8, :] = corners2D_gt[8, :]
    return corners2D_gt_corrected

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes0(output, conf_thresh, num_classes, num_keypoints=9, only_objectness=1, validation=False):

    # Parameters
    anchor_dim = 1
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    # print('output.size(1)', output.size(1))
    # print('(2*num_keypoints+1+num_classes)*anchor_dim', (2*num_keypoints+1+num_classes)*anchor_dim)
    assert(output.size(1) == (2*num_keypoints+1+num_classes)*anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -sys.maxsize
    output    = output.view(batch*anchor_dim, 2*num_keypoints+1+num_classes, h*w).transpose(0,1).contiguous().view(2*num_keypoints+1+num_classes, batch*anchor_dim*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()

    xs = list()
    ys = list()
    xs.append(torch.sigmoid(output[0]) + grid_x)
    ys.append(torch.sigmoid(output[1]) + grid_y)
    for j in range(1,num_keypoints):
        xs.append(output[2*j + 0] + grid_x)
        ys.append(output[2*j + 1] + grid_y)
    det_confs = torch.sigmoid(output[2*num_keypoints])
    cls_confs = torch.nn.Softmax()(Variable(output[2*num_keypoints+1:2*num_keypoints+1+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    for j in range(num_keypoints):
        xs[j] = convert2cpu(xs[j])
        ys[j] = convert2cpu(ys[j])
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        max_conf = conf
                        bcx = list()
                        bcy = list()
                        for j in range(num_keypoints):
                            bcx.append(xs[j][ind])
                            bcy.append(ys[j][ind])
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = list()
                        for j in range(num_keypoints):
                            box.append(bcx[j]/w)
                            box.append(bcy[j]/h)
                        box.append(det_conf)
                        box.append(cls_max_conf)
                        box.append(cls_max_id)
                        boxes.append(box)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return boxes

def get_region_boxes(output, num_classes, num_keypoints, only_objectness=1, validation=False):

    # Parameters
    anchor_dim = 1
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    # print('output.size(1)', output.size(1))
    # print('(2*num_keypoints+1+num_classes)*anchor_dim', (2*num_keypoints+1+num_classes)*anchor_dim)
    assert(output.size(1) == (2*num_keypoints+1+num_classes)*anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    max_conf = -sys.maxsize
    output = output.view(batch*anchor_dim, 2*num_keypoints+1+num_classes, h*w).transpose(0,1).contiguous().view(2*num_keypoints+1+num_classes, batch*anchor_dim*h*w)
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()

    xs = list()
    ys = list()
    xs.append(torch.sigmoid(output[0]) + grid_x)
    ys.append(torch.sigmoid(output[1]) + grid_y)
    for j in range(1,num_keypoints):
        xs.append(output[2*j + 0] + grid_x)
        ys.append(output[2*j + 1] + grid_y)
    det_confs = torch.sigmoid(output[2*num_keypoints])
    cls_confs = torch.nn.Softmax()(Variable(output[2*num_keypoints+1:2*num_keypoints+1+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    for j in range(num_keypoints):
        xs[j] = convert2cpu(xs[j])
        ys[j] = convert2cpu(ys[j])
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > max_conf:
                        max_conf = conf
                        bcx = list()
                        bcy = list()
                        for j in range(num_keypoints):
                            bcx.append(xs[j][ind])
                            bcy.append(ys[j][ind])
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = list()
                        for j in range(num_keypoints):
                            box.append(bcx[j]/w)
                            box.append(bcy[j]/h)
                        box.append(det_conf)
                        box.append(cls_max_conf)
                        box.append(cls_max_id)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return box

def get_region_boxes2(output, conf_thresh, num_classes, only_objectness=1, validation=False):

    # Parameters
    anchor_dim = 1
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (19+num_classes)*anchor_dim)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -100000
    output    = output.view(batch*anchor_dim, 19+num_classes, h*w).transpose(0,1).contiguous().view(19+num_classes, batch*anchor_dim*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*anchor_dim, 1, 1).view(batch*anchor_dim*h*w).cuda()
    xs0       = torch.sigmoid(output[0]) + grid_x
    ys0       = torch.sigmoid(output[1]) + grid_y
    xs1       = output[2] + grid_x
    ys1       = output[3] + grid_y
    xs2       = output[4] + grid_x
    ys2       = output[5] + grid_y
    xs3       = output[6] + grid_x
    ys3       = output[7] + grid_y
    xs4       = output[8] + grid_x
    ys4       = output[9] + grid_y
    xs5       = output[10] + grid_x
    ys5       = output[11] + grid_y
    xs6       = output[12] + grid_x
    ys6       = output[13] + grid_y
    xs7       = output[14] + grid_x
    ys7       = output[15] + grid_y
    xs8       = output[16] + grid_x
    ys8       = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > max_conf:
                        max_conf = conf
                        max_ind = ind

                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
            # if len(boxes) == 0:
            #     bcx0 = xs0[max_ind]
            #     bcy0 = ys0[max_ind]
            #     bcx1 = xs1[max_ind]
            #     bcy1 = ys1[max_ind]
            #     bcx2 = xs2[max_ind]
            #     bcy2 = ys2[max_ind]
            #     bcx3 = xs3[max_ind]
            #     bcy3 = ys3[max_ind]
            #     bcx4 = xs4[max_ind]
            #     bcy4 = ys4[max_ind]
            #     bcx5 = xs5[max_ind]
            #     bcy5 = ys5[max_ind]
            #     bcx6 = xs6[max_ind]
            #     bcy6 = ys6[max_ind]
            #     bcx7 = xs7[max_ind]
            #     bcy7 = ys7[max_ind]
            #     bcx8 = xs8[max_ind]
            #     bcy8 = ys8[max_ind]
            #     cls_max_conf = cls_max_confs[max_ind]
            #     cls_max_id = cls_max_ids[max_ind]
            #     det_conf =  det_confs[max_ind]
            #     box = [bcx0/w, bcy0/h, bcx1/w, bcy1/h, bcx2/w, bcy2/h, bcx3/w, bcy3/h, bcx4/w, bcy4/h, bcx5/w, bcy5/h, bcx6/w, bcy6/h, bcx7/w, bcy7/h, bcx8/w, bcy8/h, det_conf, cls_max_conf, cls_max_id]
            #     boxes.append(box)
            #     all_boxes.append(boxes)
            # else:
            #     all_boxes.append(boxes)

        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return boxes

def read_truths(lab_path, num_keypoints=9):
    num_labels = 2*num_keypoints+3 # +2 for width, height, +1 for class label
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//num_labels, num_labels) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, num_keypoints=9):
    num_labels = 2 * num_keypoints + 1
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        for j in range(num_labels):
            new_truths.append(truths[i][j])
    return np.array(new_truths)

def read_pose(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # truths = truths.reshape(truths.size/21, 21) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets

def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))

    idx = 18
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][idx]

    # _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        for j in range(i+1, len(boxes)):
            box_j = boxes[j]
            iou = bbox_iou(box_i, box_j)
            if iou > nms_thresh:
                # print('IOU', iou)
                out_boxes.append(box_i)
                box_j = 0
    # print('found boxes, after further applying NMS boxes', len(out_boxes))
    return out_boxes

def bbox_iou(box1, box2, x1y1x2y2=False):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        my = min(box1[1], box2[1])
        Mx = max(box1[2], box2[2])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def nmsv2(dets, nms_thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


# remove multiple indices from a list
# axesList = [i for j, i in enumerate(axesList) if j not in indices]
