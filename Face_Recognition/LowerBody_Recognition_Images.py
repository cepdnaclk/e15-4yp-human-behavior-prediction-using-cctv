import cv2
import os

cascade = cv2.CascadeClassifier('../Assets/Cascades/haarcascade_lowerbody.xml');

path = '../Assets/FootImages'
fileNameList = os.listdir(path)
for fileName in fileNameList:
    img = cv2.imread(f'{fileDir}/{fileName}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    body = cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in body:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Upper Body',img)
    cv2.waitKey(2000)

cv2.destroyAllWindows()