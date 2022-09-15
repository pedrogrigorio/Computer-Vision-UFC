import sys
import cv2
import numpy as np

# read image and convert to HSV
filename = sys.argv[1]
img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# create masks
lower_blue = np.array([70, 50, 40])
upper_blue = np.array([110, 255, 255])
mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

lower_green = np.array([20,50,20])
upper_green = np.array([70,255,255])
mask2 = cv2.inRange(hsv, lower_green, upper_green)

# merge masks
mask = cv2.bitwise_or(mask1, mask2)

# invert the mask
mask_inv = 255 - mask

# apply masks
target = cv2.bitwise_and(img,img, mask=mask)
background = cv2.bitwise_and(img, img, mask=mask_inv)

# exchange green and blue channel
b,g,r = cv2.split(target)
target_inv = cv2.merge((g,b,r))

# merge new target with background
result = cv2.add(target_inv, background)

# show the images
cv2.imshow('Imagem base', img)
cv2.imshow('Characters_mask', mask)
cv2.imshow('Background_mask', mask_inv)
cv2.imshow('Target', target)
cv2.imshow('Colors Changed', target_inv)
cv2.imshow('Result', result)

cv2.imwrite('Result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()