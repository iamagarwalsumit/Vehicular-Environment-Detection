import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
from skimage import filter
from skimage import color
from skimage import io
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

filename = "road2.jpg"
#im = mpimg.imread(filename)
img = cv2.imread(filename,0)
rows,cols = img.shape[:2]
#cv2.imshow("img",img)
#cv2.waitKey(0)
#img = rgb2gray(img)
#gray= rgb2gray(im)
#gray1 = ndi.gaussian_filter(gray, 4)
#edges = filter.canny(gray1)

blur = cv2.GaussianBlur(img,(7,7),0)
edge2 = cv2.Canny(blur,100,200)
#lines = probabilistic_hough_line(edges, threshold=10, line_length=40,
 #                                line_gap=3)
contours,hierarchy = cv2.findContours(edge2[rows/2:rows,:cols/2],1,2)
cnt = contours[0]

[vx,vy,x,y] = cv2.fitLine(cnt, cv2.cv.CV_DIST_L2,0,0.01,0.01)
hull = cv2.convexHull(cnt)
print(hull)
#print(vx,vy,x,y)
lefty = int((-x*vy/vx) + y+rows/2)
righty = int(((cols-x)*vy/vx)+y+rows/2)

cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
for i in range(len(hull)-1):
	cv2.line(img,(hull[i].item(1),hull[i].item(0)),(hull[i+1].item(1),hull[i+1].item(0)),(255,0,0),2)

contours,hierarchy = cv2.findContours(edge2[rows/2:rows,cols/2:cols-1],1,2)
cnt = contours[0]

[vx,vy,x,y] = cv2.fitLine(cnt, cv2.cv.CV_DIST_L2,0,0.01,0.01)

lefty = int((-(x+cols/2)*vy/vx) + y+rows/2)
righty = int(((cols-(x+cols/2))*vy/vx)+y+rows/2)
cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(blur, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Grey Scale gaussian filter', fontsize=20)

ax3.imshow(edge2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()
