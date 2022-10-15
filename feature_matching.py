import numpy as np
import cv2 as cv
from scipy import fftpack
import datetime

MIN_MATCH_COUNT = 10

# img1 = cv.imread(root_dir + 'template.jpg', 0) # 无人机图像
# img2 = cv.imread(root_dir + 'scene.jpg', 0) # 卫星图像

# ORB
def orb_matching(img1, img2):
    # 初始化ORB检测器
    orb = cv.ORB_create()
    # 基于ORB找到关键点和检测器
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 创建BF匹配器的对象
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # 匹配描述符.
    # matches = bf.knnMatch(des1,des2,k=2) # 根据距离排序
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # matches = sorted(matches, key = lambda x:x.distance) # 绘制前10的匹配项
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # 用绿色绘制匹配
                       singlePointColor=None,
                       matchesMask=matchesMask,  # 只绘制内部点
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img2
    # plt.imshow(img3, cmap='gray'),plt.show()

# sift
def sift_matching(img1, img2):
    # 初始化SIFT描述符
    sift = cv.xfeatures2d.SIFT_create()
    # 用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # ＃根据Lowe的比率测试存储所有符合条件的匹配项。
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,c = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # 用绿色绘制匹配
                       singlePointColor = None,
                       matchesMask = matchesMask, # 只绘制内部点
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img2, dst, M
    # plt.imshow(img3, cmap='gray'),plt.show()

# surf
def surf_matching(img1, img2):
    # 初始化SURF描述符
    minHessian = 400
    surf = cv.xfeatures2d.SURF_create(minHessian)
    # 用SIFT找到关键点和描述符
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    # FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 16)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params) # flann：Fast Library forApproximate Nearest Neighbors
    matches = flann.knnMatch(des1, des2, k=2)
    # ＃根据Lowe的比率测试存储所有符合条件的匹配项。
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
    # print(len(good))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3) # 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列），使用最小均方误差或者RANSAC方法
        matchesMask = mask.ravel().tolist()
        h,w,c = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,10, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # 用绿色绘制匹配
                       singlePointColor = None,
                       matchesMask = matchesMask, # 只绘制内部点
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img2, img3, dst, M

def akaze_matching(img1, img2):
    akaze = cv.AKAZE_create()
    # 用SIFT找到关键点和描述符
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    # FLANN匹配器
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = bf.knnMatch(des1, des2, k=2)
    # ＃根据Lowe的比率测试存储所有符合条件的匹配项。
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # 用绿色绘制匹配
                       singlePointColor=None,
                       matchesMask=matchesMask,  # 只绘制内部点
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img3

def fft_matching(template, origin):
    start_time = datetime.datetime.now()

    template_gray = template
    origin_gray = origin

    # template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    t_height, t_width = template_gray.shape
    o_height, o_width = origin_gray.shape

    t_fft = fftpack.fft2(template_gray, shape=(o_height, o_width))
    o_fft = fftpack.fft2(origin_gray)

    c = np.fft.ifft2(np.multiply(o_fft, t_fft.conj()) / np.abs(np.multiply(o_fft, t_fft.conj())))
    c = c.real

    result = np.where(c == np.amax(c))
    # zip the 2 arrays to get the exact coordinates
    max_coordinate = list(zip(result[0], result[1]))[0]
    # print(max_coordinate)

    start_point = (max_coordinate[1], max_coordinate[0])
    end_point = (max_coordinate[1] + t_height, max_coordinate[0] + t_width)

    # Blue color in BGR
    color = (255, 0, 0)
    thickness = 4
    image = cv.rectangle(origin, start_point, end_point, color, thickness)

    match_time = datetime.datetime.now()

    print('matching time: ' + str((match_time - start_time).seconds) + 's')