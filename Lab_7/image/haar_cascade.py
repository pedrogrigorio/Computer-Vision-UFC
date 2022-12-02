import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Open image
filename = sys.argv[1]
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Other models:
# https://github.com/opencv/opencv/tree/master/data/haarcascades


## Load models
model_path1 = "./models/haarcascade_frontalface_default.xml"
model_path2 = "./models/haarcascade_profileface.xml"
model_path3 = "./models/haarcascade_lefteye_2splits"
model_path4 = "./models/haarcascade_righteye_2splits"
model_path5 = "./models/haarcascade_smile"
frontalface = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profileface = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
left_eye    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
smile       = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detect frontal faces
frontalface_results = frontalface.detectMultiScale(img_gray, scaleFactor=1.1,minNeighbors=6,minSize=(25,25))

# Draw
for (x,y,w,h) in frontalface_results:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

# Detect profile faces
profileface_results = profileface.detectMultiScale(img_gray, scaleFactor=1.1,minNeighbors=1,minSize=(25,25))

# Draw
for (x,y,w,h) in profileface_results:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

# Detect smiles
smile_results = smile.detectMultiScale(img_gray, scaleFactor=1.3,minNeighbors=40,minSize=(20, 20))

# Draw
for (x,y,w,h) in smile_results:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(img, 'Smile', (x,y), 1, 1, (0,0,255), 1)

# Detect left eyes
left_eye = left_eye.detectMultiScale(img_gray, scaleFactor=1.15,minNeighbors=5,minSize=(20, 20))

# Exibe os resultados
for (x,y,w,h) in left_eye:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img, 'L_eye', (x,y), 1, 1, (0,255,0), 1)

# Detect right eyes
right_eye = right_eye.detectMultiScale(img_gray, scaleFactor=1.15,minNeighbors=5,minSize=(20, 20))

# Draw
for (x,y,w,h) in right_eye:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img, 'R_eye', (x,y), 1, 1, (0,255,0), 1)

cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


