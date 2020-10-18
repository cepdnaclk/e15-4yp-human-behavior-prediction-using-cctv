import cv2 as cv
import numpy as np
import time

video = cv.VideoCapture(0)
frameRate = 20
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    check, currFrame = video.read()
    #currFrame = cv.resize(currFrame, (1920, 1080))
    #B, G, R = cv.split(currFrame)

    #intensity = 150
    #BNoise = np.uint8(np.random.randint(0, 255-intensity, size=(currFrame.shape[:2])))
    #GNoise = np.uint8(np.random.randint(0, 255-intensity, size=(currFrame.shape[:2])))
    #RNoise = np.uint8(np.random.randint(0, 255-intensity, size=(currFrame.shape[:2])))

    #BMask = (B < intensity).astype(int)
    #GMask = (G < intensity).astype(int)
    #RMask = (R < intensity).astype(int)

    #BNoise = np.uint8(np.multiply(BNoise, BMask))
    #GNoise = np.uint8(np.multiply(GNoise, GMask))
    #RNoise = np.uint8(np.multiply(RNoise, RMask))

    #B = np.uint8(np.add(B, BNoise))
    #G = np.uint8(np.add(G, GNoise))
    #R = np.uint8(np.add(R, RNoise))

    #editFrame = cv.merge([B, G, R])
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
