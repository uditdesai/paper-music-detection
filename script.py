import cv2
import numpy as np
from shapedetector import ShapeDetector
from playsound import playsound
import time


# lower and upper bound for yellow color in HSV
lowerBound = np.array([22, 63, 100])
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
points = []
# play bool
play_sound = False

# Set up opening and closing filter
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

fingerExists = False
fingerInShape = False

buffer = None

# Infinite for loop
while True:

    # read camera
    ret, img = cam.read()

    # capture first frame to recognize shapes
    if first_frame is None:
        first_frame = img
        continue

    if fingerExists == False:
        buffer = img

    # get original height and width of shape
    height, width, layers = img.shape
    # get smaller frame to capture shapes
    imgSmall = cv2.resize(buffer, (int(width/4), int(height/4)))
    # resize main frame so its easier to see
    img = cv2.resize(img, (int(width), int(height)))
    img_copy = img
    # get ratio to multiply shape contours by when drawing them
    ratio = img.shape[0] / float(imgSmall.shape[0])

    # transform smaller frame to find shape contours
    gray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # initialize shape detecting class
    sd = ShapeDetector()

    # get contours from transformed small frame
    cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_NONE)

    # loop through contours
    newShapes = []

    new_cnts = [c for c in cnts if 300 < cv2.contourArea(c) < 4000]
    new_cnts = [c for c in new_cnts if cv2.contourArea(c, True) > 0]
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
        newShapes.append(
            {"shapenum": c, "shape": new_cnts[c], "shapename": shape, "area": cv2.contourArea(new_cnts[c]), "x-co": cX})

        # make detected_shapes true to only store shapes once

    shapes = newShapes
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

    if len(conts) > 0:
        fingerExists = True
    else:
        fingerExists = False

    found = False
    currently_in = {"shape": "", "x-co": 0, "area": 0}
    starting_y = 0
    ending_y = 0
    # draw yellow color rectangle
    cv2.drawContours(img, conts, -1, (255, 0, 0), 3)

    for i in range(len(conts)):
        points.append({"innerList": []})

    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        points[i]["innerList"].append(y+h/2)

        for shape in shapes:
            if cv2.pointPolygonTest(shape["shape"], (x+w/2, y+h/2), True) >= 0 or cv2.pointPolygonTest(shape["shape"], (x, y), True) >= 0:
                fingerInShape = True
                found = True
                currently_in = {
                    "shape": shape["shapename"], "x-co": shape["x-co"], "area": shape["area"]}
                print(shape["area"])
            elif cv2.pointPolygonTest(shape["shape"], (x+w, y+h), True) >= 0:
                fingerInShape = True
                found = True
                currently_in = {
                    "shape": shape["shapename"], "x-co": shape["x-co"], "area": shape["area"]}
            else:
                if found == False:
                    fingerInShape = False

    for i in range(len(conts)):
        if len(points[i]["innerList"]) > 1:
            if points[i]["innerList"][1] - points[i]["innerList"][0] > 10:
                if fingerInShape:
                    if currently_in["shape"] == "circle" or currently_in["shape"] == "pentagon":
                        if currently_in["area"] > 36000:
                            playsound('bass.wav', False)
                        elif currently_in["area"] > 25000:
                            playsound('kick.wav', False)
                        elif currently_in["x-co"] > 650:
                            playsound('snare.mp3', False)
                        else:
                            playsound('hat.wav', False)
                    else:
                        if currently_in["x-co"] > 1050:
                            playsound('C3.mp3', False)
                        elif currently_in["x-co"] > 900:
                            playsound('D.mp3', False)
                        elif currently_in["x-co"] > 800:
                            playsound('E.mp3', False)
                        elif currently_in["x-co"] > 650:
                            playsound('F.mp3', False)
                        elif currently_in["x-co"] > 500:
                            playsound('G.mp3', False)
                        elif currently_in["x-co"] > 400:
                            playsound('A.mp3', False)
                        elif currently_in["x-co"] > 250:
                            playsound('B.mp3', False)
                        else:
                            playsound('C.mp3', False)
            points[i]["innerList"].remove(points[i]["innerList"][0])

    # show original frame with shapes and yellow objects
    resized_img = cv2.resize(img, (int(width/2), int(height/2)))
    cv2.imshow("image", resized_img)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
