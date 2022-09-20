import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = sys.argv[1]
img = cv2.imread(filename)

#dimensoes
width = img.shape[1]
height = img.shape[0]

#separando canais
b, g, r = cv2.split(img)

#question 2 (grayscale)
kernel = np.float32([[0,0,0],[0,11,0],[0,0,0]])*(1/100)
b2 = cv2.filter2D(b,-1,kernel)

kernel = np.float32([[0,0,0],[0,59,0],[0,0,0]])*(1/100)
g2 = cv2.filter2D(g,-1,kernel)

kernel = np.float32([[0,0,0],[0,30,0],[0,0,0]])*(1/100)
r2 = cv2.filter2D(r,-1,kernel)

gray = b2 + g2 + r2

#question 3 (sepia)
kernel = np.float32([[0,0,0],[0,7,0],[0,0,0]])*(1/100)
b3 = cv2.filter2D(b,-1,kernel)

kernel = np.float32([[0,0,0],[0,34,0],[0,0,0]])*(1/100)
g3 = cv2.filter2D(g,-1,kernel)

kernel = np.float32([[0,0,0],[0,43,0],[0,0,0]])*(1/100)
r3 = cv2.filter2D(r,-1,kernel)

sepia = cv2.merge((b3,g3,r3))

cv2.imshow('Gray_scale',gray)
cv2.imshow('Sepia',sepia)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()