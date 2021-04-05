from __future__ import division

from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import glob
import torch
import numpy as np
import cv2
import time
from PIL import Image

from PersonIdentification.yolo.models import *
from PersonIdentification.yolo.utils import *
from PersonIdentification.yolo.datasets import *

from torch.autograd import Variable

from pathlib import Path
import datetime

########################################################################################################################
model_def = "model_data/yolo/yolov3.cfg"
weights_path = "model_data/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416
dist_thres = 0.8
face_thres = 0.9
save_time_thres = 20 #In seconds

###################################################### YOLO ############################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

yolo = Darknet(model_def, img_size=image_size).to(device)
yolo.load_darknet_weights(weights_path)

yolo.cuda()
yolo.eval() # Set in evaluation mode
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

########################################################################################################################
video = cv2.VideoCapture(0)
if video.isOpened():
    check, currFrame = video.read()
    orig_h, orig_w = currFrame.shape[:2]

    #orig_w = int(orig_w * 0.5)
    #orig_h = int(orig_h * 0.5)

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (image_size / max(orig_h, orig_w))
    pad_y = max(orig_w - orig_h, 0) * (image_size / max(orig_h, orig_w))

    # Image height and width after padding is removed
    unpad_h = image_size - pad_y
    unpad_w = image_size - pad_x


while video.isOpened():
    check, currFrame = video.read()
    #currFrame = cv2.resize(currFrame, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
    imgTensor = transforms.ToTensor()(currFrame)
    imgTensor, _ = pad_to_square(imgTensor, 0)
    imgTensor = resize(imgTensor, 416).unsqueeze(0)
    imgTensor = Variable(imgTensor.type(Tensor))

    with torch.no_grad():
        detections = yolo(imgTensor)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        if int(cls_pred) == 0: #Only Detect Humans
            # Rescale bounding boxes to dimension of original image
            x1 = int(abs(((x1 - pad_x // 2) / unpad_w) * orig_w))
            y1 = int(abs(((y1 - pad_y // 2) / unpad_h) * orig_h))
            x2 = int(abs(((x2 - pad_x // 2) / unpad_w) * orig_w))
            y2 = int(abs(((y2 - pad_y // 2) / unpad_h) * orig_h))
            #cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
            human = currFrame[y1:y2, x1:x2]
            human = cv2.resize(human, (125, 300), interpolation=cv2.INTER_AREA)
    #######################################################################################333

    cv2.imshow('Humans', human)
    cv2.imshow('Camera', currFrame)

    torch.cuda.empty_cache()
    key = cv2.waitKey(1)
    if key == 'q':
        break

cv2.destroyAllWindows()