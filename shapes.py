import cv2
import numpy as np
from shapedetector import ShapeDetector

cam = cv2.VideoCapture(0)

ret, img = cam.read()
height, width, layers = img.shape
img = cv2.resize(img, (int(width/2), int(height/2)))
imgSmall = cv2.resize(img, (int(width/4), int(height/4)))
ratio = img.shape[0] / float(imgSmall.shape[0])

gray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]

cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)

sd = ShapeDetector()

# cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                            cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"] * ratio)
        cY = int(M["m01"] / M["m00"] * ratio)
    shape = sd.detect(c)

    # draw the contour and center of the shape on the image
    # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    # cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    # cv2.putText(img, "center", (cX - 20, cY - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

cv2.imshow("img", img)

cv2.waitKey(0)
