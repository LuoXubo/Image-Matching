import os
import cv2 as cv
import numpy as np
import datetime
from feature_matching import sift_matching, surf_matching, akaze_matching
import math

# root_dir_25 = '../LPN_result/LPN_result/result_0.25/'
# root_dir_50 = '../LPN_result/LPN_result/result_0.5/'
root_dir_25 = '../LPN/images_drone-satellite/result_0.25'
root_dir_50 = '../LPN/images_drone-satellite/result_0.5/'
scene = cv.imread('./180940307034.tif', 1)
scene_25 = cv.resize(scene, (0, 0), None, 0.25, 0.25)
scene_50 = cv.resize(scene, (0, 0), None, 0.5, 0.5)

central_coords_x = 2172
central_coords_y = 1448
pt_drone = np.matrix([int(central_coords_x / 2), int(central_coords_y / 2), 1])

def transformation(pt_drone, H):
    return H @ pt_drone

def coords(mat):
    a = mat[0][0] / mat[2][0]
    b = mat[1][0] / mat[2][0]
    return float(a), float(b)

def init_gt(fpath):
    f = open(fpath)
    lines = f.readlines()
    gt_pos = {}
    for line in lines:
        num, x, y = line.split(' ')
        num = int(num)
        gt_pos[num] = [float(x), float(y)]
    return gt_pos


def calc(root_dir, scale, method):
    start_time = datetime.datetime.now()
    c_points = {}
    for img_name in os.listdir(root_dir):
        num = int(img_name[-2:])
        img_dir = root_dir + img_name + '/'
        template = cv.imread(img_dir + os.listdir(img_dir)[0], 1)
        if scale == 0.25:
            if method == 'surf':
                res, dst, H = surf_matching(template, scene_25)
            else:
                res, dst, H = sift_matching(template, scene_25)
        else:
            if method == 'surf':
                res, dst, H = surf_matching(template, scene_50)
            else:
                res, dst, H = sift_matching(template, scene_50)

        # c_point = get_center_point(dst, scale)
        # c_points[num] = c_point

        pt_sate = transformation(pt_drone.T, H)
        x, y = coords(pt_sate)
        c_points[num] = [x, y]

        # if scale == 0.25:
        #     if method == 'surf':
        #         cv.imwrite('../LPN_result/LPN_result/matching_result_SURF_0.25/' + img_name + '.jpg', res)
        #     else:
        #         cv.imwrite('../LPN_result/LPN_result/matching_result_SIFT_0.25/' + img_name + '.jpg', res)
        # else:
        #     if method == 'surf':
        #         cv.imwrite('../LPN_result/LPN_result/matching_result_SURF_0.5/' + img_name + '.jpg', res)
        #     else:
        #         cv.imwrite('../LPN_result/LPN_result/matching_result_SIFT_0.5/' + img_name + '.jpg', res)
    matching_time = datetime.datetime.now()
    print('average time: ', (matching_time-start_time).seconds/len(os.listdir(root_dir)), 's')

    dist = []
    for i in gt_points.keys():
        coord_res = c_points[i]
        coord_gt = gt_points[i]
        coord_gt[0] *= scale
        coord_gt[1] *= scale
        curdist = math.sqrt((coord_res[0]-coord_gt[0])**2 + (coord_res[1]-coord_gt[1])**2)
        dist.append(curdist)
    print('average distance: ', str(sum(dist)/len(dist)))

print('loading ground truth points ...')
# gt_points = init_gt('../GroundTruth/GT.txt')
gt_points = init_gt('./GT.txt')
print('done ...')

print('scale = 0.25')
calc(root_dir_25, 0.25, 'surf')
calc(root_dir_25, 0.25, 'sift')
print('done ...')

print('scale = 0.50')
calc(root_dir_50, 0.50, 'surf')
calc(root_dir_50, 0.50, 'sift')
print('done...')



