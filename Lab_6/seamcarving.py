#pip install seam-carving
#https://pypi.org/project/seam-carving/

import numpy as np
from PIL import Image
import cv2
import sys
import argparse
import seam_carving

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fe","--forward", action='store_true', help="Calcular a energia adicionada")
ap.add_argument("-hf","--height_first", action='store_true', help="Remover primeiro as linhas")
ap.add_argument("-wt", "--weight", type=int,default="1920", help="Comprimento")
ap.add_argument("-ht", "--height", type=int,default="1080", help="Altura")
args = vars(ap.parse_args())

#Video
cap = cv2.VideoCapture('Yoru.mp4')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_count = 0

#Novas dimensões
weight = args["weight"]
height = args["height"]
new_size = (weight, height)
print(new_size)

out = cv2.VideoWriter('result.mp4', fourcc, 20.0, new_size)

if (cap.isOpened() == False):
    print("Error")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        #Cálculo da energia
        if(args["forward"]):
            e_mode = 'forward'
        else:
            e_mode = 'backward'

        #Ordem de remoção dos seams
        if(args["height_first"]):
            carving_order = 'height-first'
        else:
            carving_order = 'width-first'

        #Seam Carving
        frame_seam = seam_carving.resize(frame, new_size, energy_mode='forward', order=carving_order)

        frame_count += 1
        print(frame_count)
        

        out.write(frame_seam)
        cv2.imshow('Frame', frame_seam)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()