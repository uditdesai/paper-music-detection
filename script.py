import cv2
import numpy as np
from shapedetector import ShapeDetector

# lower and upper bound for yellow color in HSV
lowerBound = np.array([29, 63, 100])
upperBound = np.array([31, 255, 255])

# Get camera
cam = cv2.VideoCapture(0)
# initial frame variable
first_frame = None
# initial shape detection bool
detected_shapes = False
# list of shapes
shapes = []
# initial interaction point
init_inter_points = []

# Set up opening and closing filter
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

# Infinite for loop
while True:

    # read camera
    ret, img = cam.read()

    # capture first frame to recognize shapes
    if first_frame is None:
        first_frame = img
        continue

    # get original height and width of shape
    height, width, layers = img.shape
    # get smaller frame to capture shapes
    imgSmall = cv2.resize(first_frame, (int(width/3), int(height/3)))
    # resize main frame so its easier to see
    img = cv2.resize(img, (int(width/1.5), int(height/1.5)))
    # get ratio to multiply shape contours by when drawing them
    ratio = img.shape[0] / float(imgSmall.shape[0])

    # transform smaller frame to find shape contours
    gray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]

    # initialize shape detecting class
    sd = ShapeDetector()

    # get contours from transformed small frame
    cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    # loop through contours
    new_cnts = [c for c in cnts if 1000 < cv2.contourArea(c) < 2000]
    for c in range(len(new_cnts)):

        # compute the center of the contour
        M = cv2.moments(new_cnts[c])
        # if going to divide by 0 then skip
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"] * ratio)
            cY = int(M["m01"] / M["m00"] * ratio)
        shape = sd.detect(new_cnts[c])

        new_cnts[c] = new_cnts[c].astype("float")
        new_cnts[c] *= ratio
        new_cnts[c] = new_cnts[c].astype("int")
        cv2.drawContours(img, [new_cnts[c]], -1, (0, 255, 0), 2)
        cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # append to shapes array to save shape
        if detected_shapes == False:
            shapes.append({"shapenum": c, "shape": new_cnts[c]})

    # make detected_shapes true to only store shapes once
    detected_shapes = True

    # masking image to get yellow color
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # opens and closes logic
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    # find contours of yellow color shapes
    maskFinal = maskClose
    conts, h = cv2.findContours(
        maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw yellow color rectangle
    cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        for shape in shapes:
            if cv2.pointPolygonTest(shape["shape"], (x+w/2, y+h/2), True) >= 0:
                print((x+w/2, y+h/2, shape["shapenum"]))

    # show original frame with shapes and yellow objects
    cv2.imshow("image", img)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
