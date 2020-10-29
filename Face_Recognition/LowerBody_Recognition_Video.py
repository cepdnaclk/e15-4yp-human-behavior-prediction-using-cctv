import cv2

cascade = cv2.CascadeClassifier('../Assets/Cascades/haarcascade_lowerbody.xml');

frameRate = 24  # fps
video = cv2.VideoCapture('../Assets/Videos/walk4_fast.mp4')

while True:
    check, currFrame = video.read()
    if check:
        currFrame = cv2.resize(currFrame, (960, 540))
        gray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)

        body = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in body:
            cv2.rectangle(currFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Web Cam", currFrame)
        key = cv2.waitKey(int(1000 / frameRate))
    else:
        break

video.release()
cv2.destroyAllWindows()