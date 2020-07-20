import cv2 as cv
import glob

images = glob.glob("../Images/*.jpg")

for imageName in images:
    image = cv.imread(imageName, 0)
    re = cv.resize(image, (100,100))
    cv.imshow("image", re)
    cv.waitKey(500)
    cv.destroyAllWindows()
    cv.imwrite("resized_"+imageName+".jpg", re)


