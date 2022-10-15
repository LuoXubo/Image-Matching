import numpy as npy
import cv2 as cv


root_dir = '../LPN_result/LPN_result/result_0.25/'

scene = cv.imread(root_dir + 'IMG_0369/query.jpg', 0)

scene_np = npy.array(scene)
print(scene_np)