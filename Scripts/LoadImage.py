import cv2 as cv
import numpy as np

#print(cv.__version__)
image = cv.imread('../Images/Lighthouse.jpg')
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
B, G, R = cv.split(image)

zeros = np.zeros(image.shape[:2], dtype='uint8')
fills = 255* np.ones(image.shape[:2], dtype='uint8')

intensity = 40

BNoise = np.uint8(np.random.randint(0, 255-intensity, size=(image.shape[:2])))
GNoise = np.uint8(np.random.randint(0, 255-intensity, size=(image.shape[:2])))
RNoise = np.uint8(np.random.randint(0, 255-intensity, size=(image.shape[:2])))

BMask = (B < intensity).astype(int)
GMask = (G < intensity).astype(int)
RMask = (R < intensity).astype(int)

BNoise = np.uint8(np.multiply(BNoise, BMask))
GNoise = np.uint8(np.multiply(GNoise, GMask))
RNoise = np.uint8(np.multiply(RNoise, RMask))

B = np.uint8(np.add(B, BNoise))
G = np.uint8(np.add(G, GNoise))
R = np.uint8(np.add(R, RNoise))

editFrame = cv.merge([B,G,R])
#editFrame = cv.GaussianBlur(editFrame, (5, 5), 0)
cv.imshow("HSV", hsv_image)
cv.imshow("H", hsv_image[:, :, 0])
cv.imshow("S", hsv_image[:, :, 1])
cv.imshow("V", hsv_image[:, :, 2])
#cv.imshow("Blue", cv.merge([B,zeros,zeros]))
#cv.imshow("Green", G)
#cv.imshow("Red", R)

devide = 1
resized_image = cv.resize(image,( int(image.shape[1]/devide),int(image.shape[0]/devide)))
#cv.imshow("Galaxy", resized_image)
#cv.imwrite("Galaxy_Resized.jpg", resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
