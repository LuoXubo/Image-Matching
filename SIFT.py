import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

root_dir = './datas/desk/'
img = cv.imread(root_dir+'scene.jpg')

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)
img=cv.drawKeypoints(gray,kp,img)

# cv.imshow('gray', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.imwrite(root_dir + 'sift_keypoints.jpg',img)

plt.imshow(img, cmap='gray')
plt.show()
