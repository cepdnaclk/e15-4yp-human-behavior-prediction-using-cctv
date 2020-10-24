import cv2 as cv

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv.imread("../Images/news.jpg")
greyImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(greyImage,scaleFactor=1.1,minNeighbors= 5)


for x,y,w,h, in faces:
    image = cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)


devide = 1
resized_image = cv.resize(image,( int(image.shape[1]/devide),int(image.shape[0]/devide)))
cv.imshow("Face Detection", resized_image)
cv.waitKey(0)
cv.destroyAllWindows()