import cv2 as cv
from feature_matching import surf_matching

root_dir = './datas/drone_satellite/'
scene = cv.imread(root_dir+'scene.tif', 1)
scene = cv.resize(scene, (0, 0), None, 0.25, 0.25)

template = cv.imread('Rank01-703_6950_13031.jpg', 1)

img2, img3, dst, M = surf_matching(template, scene)

cv.imwrite('./gt25.jpg', img2)
cv.imwrite('./matches25.jpg', img3)

