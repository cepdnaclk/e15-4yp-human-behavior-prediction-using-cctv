import cv2 as cv

image = cv.imread('galaxy.jpg',0)

print(type(image))
print(image)
print(image)
print(image.shape)
print(image.ndim)

resized_image = cv.resize(image,( 1920,1080))
cv.imshow("Galaxy", resized_image)
cv.waitKey(0)
#cv.destroyAllWindows()
