# some tests on tomatoes

import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 300)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 20, 255, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 40, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)  # 0 7 168 255 95 255

cap = cv2.VideoCapture('tom.mp4')
# cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outpy.avi', fourcc, 10, (frame_width, frame_height))

while (cap.isOpened()):

    # while True:
    _, frame = cap.read()
    # blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=9)
    sure_bg = cv2.dilate(mask, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 2 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [255, 0, 0]
    markers = markers.astype(np.uint8)

    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgCanny = cv2.Canny(mask, 50, 50)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        areaString = str(area)
        if area > 1000:
            cv2.drawContours(mask, contour, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(contour, True)
            # print(peri)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor > 4:
                objectType = "Pomodoro"
            else:
                objectType = "None"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(mask2, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, objectType,
                        (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)
            cv2.putText(imgCanny, objectType,
                        (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(mask2, "Area: " + areaString,
                        (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Maskcanny", imgCanny)

    imgBlank = np.zeros_like(frame)
    imgStack = stackImages(0.4, ([frame, mask2],
                                 [markers, unknown]))

    cv2.imshow("Stack", imgStack)
    out.write(imgStack)

    key = cv2.waitKey(400)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
