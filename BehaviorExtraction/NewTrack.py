from __future__ import division
import cv2
import numpy as np
import torch

########################################################################################################################
from yolo2.models import *
from yolo2.utils import *
from yolo2.datasets import *
from torch.autograd import Variable

model_def = "model_data_new/yolo/yolov3.cfg"
weights_path = "model_data_new/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416
dist_thres = 0.8
face_thres = 0.9
save_time_thres = 20 #In seconds

########################################################################################################################
from BehaviorExtraction.reid.extractor import HumanReIdentifier
from torchvision import transforms

data_path = 'database/2021-02-05'
REID_MODEL_PATH = 'model_data/ft_ResNet50_ReID/net_last.pth'
REID_FEAT_MAT = 'database/2021-02-05/features.mat'


def draw_temp_boxes(image, tracked_bboxes, names):
    for box, name in zip(tracked_bboxes, names):
        x1, y1, x2, y2 = box
        cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0] , 2)

        (text_width, text_height) , baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [255, 0, 0] , thickness=cv2.FILLED)
        cv2.putText(image, name, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [255, 255, 255], 1, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    ###################################################### YOLO ########################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    yolo = Darknet(model_def, img_size=image_size).to(device)
    yolo.load_darknet_weights(weights_path)

    yolo.cuda()
    yolo.eval() # Set in evaluation mode
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    ###################################################### ReID ########################################################
    reIdentifier = HumanReIdentifier(REID_MODEL_PATH, REID_FEAT_MAT)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ################################################## OPEN POSE #######################################################
    #skeleton_detector = SkeletonDetector(OPENPOSE_MODEL_PATH, OPENPOSE_IMG_SIZE)
    #action_classifier = MultiPersonClassifier(ACTION_MODEL_PATH, ACTION_CLASSES)

    video = cv2.VideoCapture("./assets/HumanVideo.mp4")
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

        human_crops = []
        human_boxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if int(cls_pred) == 0: #Only Detect Humans
                # Rescale bounding boxes to dimension of original image
                x1 = int(abs(((x1 - pad_x // 2) / unpad_w) * orig_w))
                y1 = int(abs(((y1 - pad_y // 2) / unpad_h) * orig_h))
                x2 = int(abs(((x2 - pad_x // 2) / unpad_w) * orig_w))
                y2 = int(abs(((y2 - pad_y // 2) / unpad_h) * orig_h))

                human = currFrame[y1:y2, x1:x2]
                human_crops.append(data_transform(human))
                human_boxes.append([x1, y1, x2, y2])

        if human_crops:
            tagNames = reIdentifier.getHumanTags(human_crops)
            currFrame = cv2.cvtColor(currFrame, cv2.COLOR_RGB2BGR)
            draw_temp_boxes(currFrame, human_boxes, tagNames)

        cv2.imshow('Camera', currFrame)
        torch.cuda.empty_cache()
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cv2.destroyAllWindows()