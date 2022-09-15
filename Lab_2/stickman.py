from doctest import NORMALIZE_WHITESPACE
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load images
circle_src = cv2.imread('./lab02/imagens/circle.jpg')
line_src = cv2.imread('./lab02/imagens/line.jpg')

# invert color
circle_mask = 255 - circle_src
line_mask = 255 - line_src

# dimensions
width_src = line_src.shape[1]
height_src = line_src.shape[0]

# new dimensioens
width = 3*width_src
height = 3*height_src

# change scale
M_scaling = np.float32([[1,0,3],[0,1,3]])
circle = cv2.warpAffine(circle_mask,M_scaling,(width,height))
line = cv2.warpAffine(line_mask,M_scaling,(width,height))

# center objects
M_line_translation = np.float32([[1,0,100],[0,1,100]])
line = cv2.warpAffine(line,M_line_translation,(width,height))

# position the head
M_head_translation = np.float32([[1,0,100],[0,1,32]])
head = cv2.warpAffine(circle,M_head_translation,(width,height))

# take the center of the image
x_center = width/2
y_center = height/2

# rotate the line to make the body.
M_line_rotation = cv2.getRotationMatrix2D((x_center,y_center),90,1)
body = cv2.warpAffine(line,M_line_rotation,(width,height))

## arms
# resize arm line
arm_line = cv2.resize(line, (0,0), fx=0.75, fy=0.75)
M_arm_scaling = np.float32([[1,0,3],[0,1,3]])
arm_line = cv2.warpAffine(arm_line, M_arm_scaling, (width,height))

# position left arm
M_arm_translation = np.float32([[1,0,2],[0,1,0]])
arm_l = cv2.warpAffine(arm_line,M_arm_translation,(width,height))

# position right arm
arm_r = cv2.flip(arm_l, 1)
M_arm_translation = np.float32([[1,0,5],[0,1,0]])
arm_r = cv2.warpAffine(arm_r,M_arm_translation,(width,height))
arms = cv2.add(arm_l, arm_r)

# legs
# resize leg line
leg_line = cv2.resize(line, (0,0), fx=1.5, fy=1.5)
M_leg_scaling = np.float32([[1,0,3],[0,1,3]])
leg_line = cv2.warpAffine(leg_line, M_leg_scaling, (width,height))

# position left leg
M_leg_translation = np.float32([[1,0,-168],[0,1,-55]])
leg_line_l = cv2.warpAffine(leg_line,M_leg_translation,(width,height))

# rotate leg
M_leg_rotation = cv2.getRotationMatrix2D((x_center,y_center),45,1)
leg_l = cv2.warpAffine(leg_line_l,M_leg_rotation,(width,height))

# position right leg
leg_r = cv2.flip(leg_l, 1)
M_leg_r_translation = np.float32([[1,0,1],[0,1,0]])
leg_r = cv2.warpAffine(leg_r,M_leg_r_translation,(width,height))

# merge legs
legs = cv2.add(leg_l, leg_r)

# position details
M_legs_translation = np.float32([[1,0,2],[0,1,3]])
legs = cv2.warpAffine(legs,M_legs_translation,(width,height))

# merge all parts
head_body = cv2.add(head, body)
arms_legs = cv2.add(arms, legs)
result = cv2.add(head_body, arms_legs)

# center the stickmanboen and invert color
M_stickman_translation = np.float32([[1,0,0],[0,1,-20]])
stickman = cv2.warpAffine(result,M_stickman_translation,(width,height))
stickman = 255 - stickman

cv2.imwrite('Stickman.jpg', stickman)
cv2.imshow('Stickman', stickman)
cv2.waitKey(0)
cv2.destroyAllWindows()