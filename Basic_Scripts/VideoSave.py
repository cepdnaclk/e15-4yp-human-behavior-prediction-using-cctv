import cv2 as cv
import pandas as pd
from datetime import datetime

frameRate = 20  # fps
firstFrame = None
video = cv.VideoCapture('../Assets/Videos/walk4_slow.mp4')

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('walk.avi', fourcc, 20.0, (640, 480))

startTime = None
while True:
    check, currFrame = video.read()
    if check:
        editFrame = cv.cvtColor(currFrame, cv.COLOR_BGR2GRAY)
        editFrame = cv.GaussianBlur(editFrame, (21, 21), 0)

        if firstFrame is None:
            firstFrame = editFrame
            continue

        editFrame = cv.absdiff(editFrame, firstFrame)
        editFrame = cv.threshold(editFrame, 30, 255, cv.THRESH_BINARY)[1]
        editFrame = cv.dilate(editFrame, None, iterations = 2)

        (contours, _) = cv.findContours(editFrame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) > 10000:
                currStatus = 1
                (x, y, w, h) = cv.boundingRect(contour)
                dilateFrame = cv.rectangle(currFrame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv.putText(currFrame, str(datetime.now()), (10, 470), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        out.write(currFrame)
        cv.imshow("Web Cam", currFrame)
        key = cv.waitKey(int(1000 / frameRate))
    else:
        break

video.release()
out.release()
cv.destroyAllWindows()