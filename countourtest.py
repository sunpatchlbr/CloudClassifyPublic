import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.pyrDown(cv2.imread('../../Data/TestPhotos/BackgroundTest/multi/multi2.JPG', cv2.IMREAD_UNCHANGED))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
sure_bg = cv2.dilate(opening, kernel, iterations=1)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1.
markers += 1

# Label the unknown region as 0.
markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers==-1] = [0,150,255]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
