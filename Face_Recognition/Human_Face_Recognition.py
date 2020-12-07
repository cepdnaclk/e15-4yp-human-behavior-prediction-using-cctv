import cv2 as cv
import numpy as np
import face_recognition as face
import pickle
import dlib
import os
import time

#####################################################################################
scale = 0.5
threshold = 0.4
#####################################################################################
def humanRecognize(path):

    print(f'GPU Usage:{dlib.DLIB_USE_CUDA}')
    print("[INFO]: Start Loading Encodes")
    encodeFileList = os.listdir(f'{path}/Encodes')
    encodeList = []
    encodeNameList = []

    for encodeFile in encodeFileList:
        encodeList.append(pickle.loads(open(f'{path}/Encodes/{encodeFile}', "rb").read()))
        encodeNameList.append(os.path.splitext(encodeFile)[0])

    print("[INFO]: Finish Loading Encodes")

    print("[INFO]: Start Video Capture")
    video = cv.VideoCapture(0) #'../Assets/Videos/Face_Video_2.mp4') #'../Assets/Videos/walk8_slow.mp4')

    while True:
        check, currFrame = video.read()
        if not check:
            print("Video Stopped")
            break

        t = time.time()
        editFrame = cv.resize(currFrame, (0, 0), cv.INTER_AREA, scale, scale)
        editFrame = cv.cvtColor(editFrame,cv.COLOR_BGR2RGB) #gpu

        currFrameFaces = face.face_locations(editFrame)
        currFrameEncodings = face.face_encodings(editFrame)

        for encodeFace, faceLocation in zip(currFrameEncodings, currFrameFaces):
            name = "Unknown"
            for i in range(len(encodeList)):
                matches = face.compare_faces(encodeList[i]['TagImageEncodes'], encodeFace)
                faceDistance = face.face_distance(encodeList[i]['TagImageEncodes'], encodeFace)
                #print(faceDistance)
                matchIndex = np.argmin(faceDistance)
                #print(f'minIndex:{matchIndex}, min:{faceDistance[matchIndex]}')

                if faceDistance[matchIndex] < threshold and matches[matchIndex]:
                    name = encodeNameList[i].upper()

            y1, x2, y2, x1 = [x * int(1/scale) for x in faceLocation]
            cv.rectangle(currFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(currFrame, name, (x1 + 6, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        fps = round(1 / (time.time() - t), 2)
        cv.putText(currFrame, f'FPS:{fps}', (6, 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.imshow("Face Recognition", currFrame)
        key = cv.waitKey(1)
        if key == 27:
            break

    video.release()
    cv.destroyAllWindows()

#####################################################################################
path = '../Assets/FaceImages'
humanRecognize(path)


