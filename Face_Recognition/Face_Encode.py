import os
import cv2 as cv
import pickle
import face_recognition as face
import time

################################################################################################
def encode(path):
    tagNameList = os.listdir(path)
    if 'Encodes' in tagNameList:
        tagNameList.remove('Encodes')
    else:
        os.mkdir(f'{path}/Encodes')

    for tagName in tagNameList:
        tagImageNameList = os.listdir(f'{path}/{tagName}')
        tagImageEncodeList = []
        for tagImageName in tagImageNameList:
            tagImage = cv.imread(f'{path}/{tagName}/{tagImageName}')
            tagImage = cv.cvtColor(tagImage, cv.COLOR_BGR2RGB)
            tagImageEncodes = face.face_encodings(tagImage)
            if (len(tagImageEncodes) >= 1):
                tagImageEncodeList.append(tagImageEncodes[0]) #save only one image

        print("[INFO] Serialize {}".format(tagName))
        data = {"TagImageEncodes": tagImageEncodeList, "TagImageName": tagImageNameList}
        f = open(f'{path}/Encodes/{tagName}.pickle', "wb")
        f.write(pickle.dumps(data))
################################################################################################

path = '../Assets/FaceImages'

t = time.time()
encode(path)
print(f'Elapsed Time:{round(time.time()-t, 2)}s')
