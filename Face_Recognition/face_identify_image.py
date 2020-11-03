import cv2 as cv
import face_recognition as face
import dlib
print(dlib.DLIB_USE_CUDA)

imgElon = face.load_image_file('../Assets/FaceImages/musk/musk0.jpg')
imgElon = cv.cvtColor(imgElon,cv.COLOR_BGR2RGB)


imgTest = face.load_image_file('../Assets/FaceImages/musk/musk3.jpg')
imgTest = cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

faceLoc = face.face_locations(imgElon)[0]
encodeElon = face.face_encodings(imgElon)[0]
imgElon = cv.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 3)

faceLocTest = face.face_locations(imgTest)[0]
encodeTest = face.face_encodings(imgTest)[0]
imgTest = cv.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 0), 3)

results = face.compare_faces([encodeElon], encodeTest)
faceDistance = face.face_distance([encodeElon], encodeTest)

cv.putText(imgTest, f'{results} {round(faceDistance[0],2)}', (50,50),cv.FONT_HERSHEY_COMPLEX, 1, (0,255,255),2)
cv.imshow('Elon Musk', imgElon)
cv.imshow('Elon Test', imgTest)
cv.waitKey(0)