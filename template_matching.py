import cv2 as cv

# 列表中所有的6种比较方法
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配,计算模板与某个子图的对应像素的差值平方和；最好的匹配值为0；匹配越差，匹配值越大。
# CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
# CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
# CV_TM_SQDIFF_NORMED 归一化平方差匹配法
# CV_TM_CCORR_NORMED 归一化相关匹配法,去除了亮度线性变化对相似度计算的影响。可以保证图像和模板同时变量或变暗k倍时结果不变
# CV_TM_CCOEFF_NORMED 归一化相关系数匹配法，一般情况下用的最多吧, 把图像和模板都减去了各自的平均值，再各自除以各自的方差，保证图像和模板分别改变光照不影响计算结果，计算出的相关系数限制在-1到1之间，1 表示完全相同，-1 表示两幅图像的亮度正好相反，0 表示没有线性关系

def template_matching(scene, template, method):
    # method = eval(meth)
    w, h, c = template.shape
    # 应用模板匹配
    # start_time = datetime.datetime.now()
    res = cv.matchTemplate(scene, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(scene, top_left, bottom_right, 255, 2)
    # match_time = datetime.datetime.now()
    # print('matching time: ' + str((match_time - start_time).seconds) + 's')
    return scene
