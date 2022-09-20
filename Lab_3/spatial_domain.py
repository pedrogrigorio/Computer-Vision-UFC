import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

filename = sys.argv[1]
img = cv2.imread(filename,0)

#media 5x5
avg_blur = cv2.blur(img,(5,5))

#gaussian
gaussian_blur = cv2.GaussianBlur(img,(5,5),1)

#mediana
median_blur = cv2.medianBlur(img,5)

#image plot
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(median_blur, cmap = 'gray')
plt.title('Mediana'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(avg_blur, cmap= 'gray')
plt.title('Media'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(gaussian_blur, cmap = 'gray')
plt.title('Gaussiana'), plt.xticks([]), plt.yticks([])
plt.show()