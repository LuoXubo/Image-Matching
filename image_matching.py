# goal: image matching
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
ratio = 0.9

# 在2中找出1来

root_dir = './datas/stones/'
img1 = cv2.imread(root_dir + 'img1.jpg')
img2 = cv2.imread(root_dir + 'img2.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

start_time = datetime.datetime.now()

# calculate key points and descriptors using SIFT method
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params =dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []

for m1, m2 in matches:
    if m1.distance < ratio * m2.distance:
        good_matches.append(m1)

if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 利用cv.findHomography计算单应矩阵，筛选去除false-positive样本
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # 利用cv.perspectiveTransform对匹配框进行透视变换
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    kp_time = datetime.datetime.now()
    matchesMask = None

plt.imshow(img2)
plt.savefig(root_dir+'matchingPatch.jpg', dpi=300)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)


end_time = datetime.datetime.now()
print('matching time: ' + str((end_time - start_time).seconds) + ' second(s).')

plt.imshow(img3, 'gray')
plt.savefig(root_dir+'featureMatching.jpg', dpi=300)