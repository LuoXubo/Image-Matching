import cv2 as cv
import datetime
from feature_matching import sift_matching, surf_matching
from template_matching import template_matching

root_dir = './datas/drone_satellite/'
ls = 15

scene = cv.imread(root_dir + 'scene.tif', 1)
scene = cv.resize(scene, (0, 0), None, 0.25, 0.25)
# h, w, c = scene.shape
# scene = scene[int(h/2):, int(w/2):, :]

template = cv.imread(root_dir + 'template.jpg', 1)
template = cv.resize(template, (0, 0), None, 0.25, 0.25)


start_time = datetime.datetime.now()
# scene = template_matching(scene, template, cv.TM_SQDIFF_NORMED)
scene, _ = surf_matching(template, scene)
match_time = datetime.datetime.now()
print('matching time: ' + str((match_time - start_time).seconds) + 's')

cv.imwrite(root_dir + 'matching_result_surf.jpg', scene)
