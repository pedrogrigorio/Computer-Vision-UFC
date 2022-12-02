import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

#Video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
cap = cv2.VideoCapture('videos/faces_360p.mp4')
out = cv2.VideoWriter('result_videos/result_360p.mp4', fourcc, 20.0, (1280, 720))

# other models:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

## load model
model_path = "./lab07/models/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(model_path)

if (cap.isOpened() == False):
    print("Error")

start_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        #Detecção
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detectMultiScale(img_gray, scaleFactor=1.15,minNeighbors=3,minSize=(100, 100), maxSize=(700,700))#, flags=cv2.CASCADE_SCALE_IMAGE)

        # Exibe os resultados
        for (x,y,w,h) in results:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

end_time = time.time()
time = end_time - start_time
print(time)
cv2.destroyAllWindows()