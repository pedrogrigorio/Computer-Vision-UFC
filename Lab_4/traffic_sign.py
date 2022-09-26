import sys
from turtle import circle
import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = sys.argv[1]
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circle_lines = img.copy()
rhombus_lines = img.copy()
octagon_lines = img.copy()

#Rhombus

edges = cv2.Canny(gray, 100, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=80, maxLineGap=8)

lines = np.delete(lines, [46, 61], 0)
for line in lines:
	x1,y1,x2,y2 = line[0]
	cv2.line(rhombus_lines, (x1,y1), (x2,y2), (255,0,0), 2)
	
hsv = cv2.cvtColor(rhombus_lines, cv2.COLOR_BGR2HSV)
rhombus_mask = cv2.inRange(hsv, (100,50,50), (130,255,255))
rhombus = cv2.bitwise_and(gray, gray, mask=rhombus_mask)

#Circle

blur = cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist=5, param1=140, param2=110, minRadius=35)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
	cv2.circle(circle_lines, (i[0], i[1]), i[2], (255,0,0), 2)
	cv2.circle(circle_lines, (i[0], i[1]), 2, (255,0,0), 3)

hsv = cv2.cvtColor(circle_lines, cv2.COLOR_BGR2HSV)
circle_mask = cv2.inRange(hsv, (100,50,50), (130,255,255))
circle = cv2.bitwise_and(gray, gray, mask=circle_mask)

#Octagon

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=6)

for line in lines:
	x1,y1,x2,y2 = line[0]
	cv2.line(octagon_lines, (x1,y1), (x2,y2), (255,0,0), 2)

hsv = cv2.cvtColor(octagon_lines, cv2.COLOR_BGR2HSV)
octagon_mask = cv2.inRange(hsv, (100,50,50), (130,255,255))
octagon = cv2.bitwise_and(gray, gray, mask=octagon_mask)

circle_result = cv2.bitwise_and(edges,edges, mask=circle_mask)
rhombus_result = cv2.bitwise_and(edges,edges, mask=rhombus_mask)

# impossivel isolar octagono do losango, HoughLine não tem parâmetro para limitar o tamanho maximo da linha
#octagon_result = 

cv2.imshow("Img", edges)
cv2.imshow("Losango", rhombus_lines)
cv2.imshow("Circle", circle_lines)
cv2.imshow("Losango result", rhombus)
cv2.imshow("Circle result", circle)
cv2.imshow("Octagon", octagon)
cv2.imshow("Edges Circle", circle_result)
cv2.imshow("Edges Rhombus", rhombus_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
