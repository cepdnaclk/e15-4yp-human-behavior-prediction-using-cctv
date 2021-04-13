from __future__ import division

from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import glob
import torch
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw
from IPython import display

from PersonIdentification.yolo.models import *
from PersonIdentification.yolo.utils import *
from PersonIdentification.yolo.datasets import *

from torch.autograd import Variable

from pathlib import Path
import datetime

########################################################################################################################
data_path = "ImageDatabase/Faces/faces.pt"
model_def = "ModelFiles/yolo/yolov3.cfg"
weights_path = "ModelFiles/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416

dist_thres = 0.8
face_thres = 0.9
save_time_thres = 6 #In seconds
###################################################### YOLO ############################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

yolo = Darknet(model_def, img_size=image_size).to(device)
yolo.load_darknet_weights(weights_path)

yolo.cuda()
yolo.eval() # Set in evaluation mode
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

########################################################################################################################
mtcnn = MTCNN(keep_all=True, device=device)  # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # initializing resnet for face img to embeding conversion

saved_data = torch.load(data_path)  # loading faces.pt file
embedding_list = saved_data[0]  # getting embedding data
tag_list = saved_data[1]  # getting list of names

human_save_path = f'ImageDatabase/Human/{datetime.date.today()}/'
human_save_count = dict.fromkeys(set(tag_list), 0)
human_save_time = dict.fromkeys(set(tag_list), None)

for tag in tag_list:
    human_save_count[tag] = len(glob.glob1(human_save_path + f'{tag}', "*.jpg"))

########################################################################################################################
def draw_temp_boxes(image, tracked_bboxes, names):
    for box, name in zip(tracked_bboxes, names):
        x1, y1, x2, y2 = box
        #cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0], 2)

        (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [255, 0, 0], thickness=cv2.FILLED)
        cv2.putText(image, name, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [255, 255, 255], 1,lineType=cv2.LINE_AA)

video = cv2.VideoCapture('test/home2.mp4')
print(video.isOpened())

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
    tic = time.time()
    ####################################################
    currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
    imgTensor = transforms.ToTensor()(currFrame)
    imgTensor, _ = pad_to_square(imgTensor, 0)
    imgTensor = resize(imgTensor, 416).unsqueeze(0)
    imgTensor = Variable(imgTensor.type(Tensor))

    with torch.no_grad():
        p = time.time()
        detections = yolo(imgTensor)
        q = time.time()
        #print(f'Yolo Time: {round(q-p,3)} ({round(1/(q-p),2)})')
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    count = True
    humanFrame = np.zeros([300, 125])
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        if int(cls_pred) == 0: #Only Detect Humans
            # Rescale bounding boxes to dimension of original image
            x1 = int(abs(((x1 - pad_x // 2) / unpad_w) * orig_w))
            y1 = int(abs(((y1 - pad_y // 2) / unpad_h) * orig_h))
            x2 = int(abs(((x2 - pad_x // 2) / unpad_w) * orig_w))
            y2 = int(abs(((y2 - pad_y // 2) / unpad_h) * orig_h))
            #cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)

            human = currFrame[y1:y2, x1:x2]
            humanPIL = Image.fromarray(human)
            p = time.time()
            face, prob = mtcnn(humanPIL, save_path='face.png', return_prob=True)
            q = time.time()
            print(f'MTCNN Time: {round(q-p,3)} ({round(1/(q-p),2)})')
            if face is not None:
                probArgMax = prob.argmax()
                if prob[probArgMax] > face_thres: # Check the probability to be a human face
                    p = time.time()
                    emb = resnet(face[probArgMax].unsqueeze(0)).detach()
                    q = time.time()
                    print(f'RESNET Time: {round(q-p,3)} ({round(1/(q-p),2)})')
                    dist_list = []  # list of matched distances, minimum distance is used to identify the person
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    minDist = min(dist_list)
                    if minDist < dist_thres:
                        idx_min = dist_list.index(minDist)
                        tag = tag_list[idx_min]
                        currTime = datetime.datetime.now()
                        if human_save_time[tag] is None or save_time_thres < (currTime - human_save_time[tag]).total_seconds():
                            human_save_path = f'ImageDatabase/Human/{datetime.date.today()}/{tag}'
                            Path(human_save_path).mkdir(parents=True, exist_ok=True)
                            human = cv2.cvtColor(human, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(human_save_path, f'{tag}{human_save_count[tag]}.jpg'), human)
                            human_save_count[tag] = human_save_count[tag] + 1
                            human_save_time[tag] = currTime
                    else:
                        tag = 'Unknown'

                    human = cv2.resize(human, (250, 600), interpolation=cv2.INTER_AREA)
                    human = cv2.cvtColor(human, cv2.COLOR_RGB2BGR)
                    cv2.putText(human, tag , (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if count:
                        humanFrame = human
                        count = False
                    else:
                        humanFrame = np.concatenate((humanFrame, human), axis=1)

    ####################################################################################################################

    toc = time.time()
    currFrame = cv2.cvtColor(currFrame, cv2.COLOR_RGB2BGR)
    cv2.putText(currFrame, f'FPS:{round(1/(toc-tic),2)}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Humans', humanFrame)
    cv2.imshow('Camera', currFrame)

    torch.cuda.empty_cache()
    key = cv2.waitKey(1)
    if key == 'q':
        break

cv2.destroyAllWindows()