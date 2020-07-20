import cv2 as cv
import time

video = cv.VideoCapture(0)
frameRate = 24
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    check, currFrame = video.read()

    greyFrame = cv.cvtColor(currFrame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(greyFrame, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h, in faces:
        currFrame = cv.rectangle(currFrame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv.imshow("Web Cam", currFrame)
    key = cv.waitKey(int(1000/frameRate))

    if key == ord('z'):
        break

video.release()
cv.destroyAllWindows()
