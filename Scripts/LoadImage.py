import cv2 as cv
import glob

image = cv.imread('../Images/profile.jpg', 0)

#print(type(image))
#print(image)
#print(image)
#print(image.shape)
#print(image.ndim)

devide = 2
resized_image = cv.resize(image,( int(image.shape[1]/devide),int(image.shape[0]/devide)))
cv.imshow("Galaxy", resized_image)
cv.imwrite("Galaxy_Resized.jpg", resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
