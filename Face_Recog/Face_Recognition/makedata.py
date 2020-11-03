import cv2 as cv
import numpy as np
import face_recognition as face
import pickle
import dlib
import os
import imutils



def makedata(frames,total):
	
	global img
	i=4
	
	#path='new'+ str(i)+ 'tag'	# Load the detector
	detector = dlib.get_frontal_face_detector()
	#detector = dlib.cnn_face_detection_model_v1(frames,1)


	# Convert image into grayscale
	gray = cv.cvtColor(src=frames, code=cv.COLOR_BGR2GRAY)

	# Use detector to find landmarks
	faces = detector(gray)
	print(faces)
	for face in faces:
	    x1 = face.left() # left point
	    y1 = face.top() # top point
	    x2 = face.right() # right point
	    y2 = face.bottom() # bottom point
	    #face_crop = gray_image[y:y+h, x:x+w]
	    # Draw a rectangle
	    nw= cv.rectangle(img=frames, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)

	# show the image
	cv.imshow(winname="Face", mat=frames)


	if faces !=():
		if total<50:
		  	p = os.path.sep.join(['new', "{}.png".format(
		  		str(total).zfill(5))])
		  				#time.sleep(2.0)
		  	#cv.imwrite(p,frames)
		  	cv.imwrite(p,nw)
		else:
			return 0
	return 0

print("[INFO] starting video stream...")
#video = cv.VideoCapture('walk4_slow.mp4')
video = cv.VideoCapture(0)
scale=0.2
total=0

while True:

	check, currFrame = video.read()
	total += 1
	if check==True:
		editFrame = cv.resize(currFrame, (0, 0), cv.INTER_AREA, scale, scale)
		editFrame = cv.cvtColor(editFrame,cv.COLOR_BGR2RGB) #gpu
		currFrameFaces = face.face_locations(editFrame)
		currFrameEncodings = face.face_encodings(editFrame)
		makedata(currFrame,total)
	else:
		print("Noooo")

	key = cv.waitKey(1)
	if key == 27:
		break

    	

video.release()
cv.destroyAllWindows()

