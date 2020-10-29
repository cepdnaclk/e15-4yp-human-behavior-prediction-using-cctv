import cv2 as cv
import pandas as pd
from datetime import datetime

frameRate = 20  # fps
firstFrame = None
video = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

faceCascade = cv.CascadeClassifier('../Assets/Cascades/haarcascade_frontalface_default.xml')
df = pd.DataFrame(columns=["Start", "End"])
startTime = None

while True:
    check, currFrame = video.read()
    currStatus = 0
    editFrame = cv.cvtColor(currFrame, cv.COLOR_BGR2GRAY)
    editFrame = cv.GaussianBlur(editFrame, (21, 21), 0)

    if firstFrame is None:
        preStatus = currStatus
        firstFrame = editFrame
        continue

    editFrame = cv.absdiff(editFrame, firstFrame)
    editFrame = cv.threshold(editFrame, 30, 255, cv.THRESH_BINARY)[1]
    editFrame = cv.dilate(editFrame, None, iterations= 2)
    
    (contours, _) = cv.findContours(editFrame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) > 10000:
            currStatus = 1
            (x,y,w,h) = cv.boundingRect(contour)
            dilateFrame = cv.rectangle(currFrame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    if preStatus == 0 and currStatus == 1:
        startTime = datetime.now()
    elif preStatus == 1 and currStatus == 0:
        df = df.append({"Start": startTime, "End": datetime.now()}, ignore_index=True)

    cv.putText(currFrame, str(datetime.now()), (10, 470), cv.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    cv.imshow("Current_Frame", currFrame)
    out.write(currFrame)
    cv.imshow("Edit_Frame", editFrame)
    key = cv.waitKey(int(1000/frameRate))

    if key == 27 or check == 0:
        if currStatus == 1:
            df = df.append({"Start": startTime, "End": datetime.now()}, ignore_index=True)
        break

    preStatus = currStatus

df.to_csv("Output/Times.csv")
video.release()
out.release()
cv.destroyAllWindows()