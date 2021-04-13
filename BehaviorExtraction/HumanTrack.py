import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np

import tensorflow as tf
from yolo.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, read_class_names
from BehaviorExtraction.Configurations import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import pickle
import struct
import socket

##############################################################################################
import torch.backends.cudnn as cudnn
from reid.model import ft_net
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import scipy.io

# Options
name = 'ft_ResNet50'
data_path = 'database/faces'
model_path = 'model_data/ft_ResNet50_ReID/net_last.pth'
mat_path = 'database/faces/features.mat'
stride = 2
batchsize = 256
nclasses = 751
score_threshold = 0.7
gpu_id = 0

##############################################################################################

def draw_temp_boxes(image, tracked_bboxes, names):
    for i in range(len(tracked_bboxes)):
        id, x1, y1, x2, y2 = tracked_bboxes[i]
        name = names[i]

        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0] , 2)

        text = f'{id} - {name}'
        (text_width, text_height) , baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [255, 0, 0] , thickness=cv2.FILLED)
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [255, 255, 255], 1, lineType=cv2.LINE_AA)

def extract_feature(model, img):
    feature = torch.FloatTensor()
    n, c, h, w = img.size()
    #print(f'n:{n}, c:{c}, h:{h}, w:{w}')
    ff = torch.FloatTensor(n, 512).zero_().cuda()
    for i in range(2):
        if (i == 1):  # Flip the image
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
            img = img.index_select(3, inv_idx)

        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    feature = torch.cat((feature, ff.data.cpu()), 0)
    return feature

if __name__ == "__main__":
    iou_threshold = 0.1 #0.45
    score_threshold = 0.3
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize yolo detection object
    yolo = Load_Yolo_model()
    NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)

    # initialize deep sort object
    encoder = gdet.create_box_encoder(SRC_DEEPSORT_MODEL_PATH, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    vid = cv2.VideoCapture(SRC_VIDEO_PATH)

    if CONNECTION_ENABLE:
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((HOST, PORT))

    ##################################### ReID ###################################################
    # Set gpu ids
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    result = scipy.io.loadmat(mat_path)
    person_feature = torch.FloatTensor(result['person_feature'])
    person_ids = result['person_ids'][0]

    model = ft_net(nclasses, stride=stride)
    model.load_state_dict(torch.load(model_path))
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    name = ['Aunty', 'Risith', 'Asith', 'Madaya', 'YellowMan', 'BlueMan']
    ##############################################################################################
    while vid.isOpened():
        check, frame = vid.read()
        key = cv2.waitKey(1)
        if not check or key == 'q':
            print("Video_Stopped")
            break

        tic = time.time()

        frame_h = int(frame.shape[0] * 0.8)
        frame_w = int(frame.shape[1] * 0.8)
        frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = image_preprocess(np.copy(original_frame), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = yolo.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, YOLO_INPUT_SIZE, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if NUM_CLASS[int(bbox[5])] in ["person"]:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            #foot_coor = np.array([np.array([[bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[3]]], dtype='float32')])
            tracked_bboxes.append([track.track_id]+[round(i) for i in track.to_tlbr().tolist()])

        if CONNECTION_ENABLE:
            data_bag = {}
            data_bag['frame'] = frame
            data_bag['boxes'] = tracked_bboxes
            data = pickle.dumps(data_bag)
            clientsocket.sendall(struct.pack("L", len(data)) + data)

        else:
            if tracked_bboxes:
                image_tensor_list = []
                for id, x1, y1, x2, y2 in tracked_bboxes:
                    #print(f'Id{id}, x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    crop_img = frame[y1:y2 , x1:x2]
                    image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    image_tensor_list.append(data_transform(image))

                stacked_tensor = torch.stack(image_tensor_list)
                with torch.no_grad():
                    feature = extract_feature(model, stacked_tensor)

                query = torch.transpose(feature, 0, 1)
                score = torch.mm(person_feature, query)
                score = score.squeeze(1).cpu()

                indexes = np.argmax(score.numpy(), axis=0)
                tagNames = []
                if type(indexes) is not np.int64:
                    scores = np.diag(score.numpy()[indexes])
                    #print(scores)
                    for i, s in zip(indexes, scores):
                        if s > score_threshold:
                            tagNames.append(name[person_ids[i]])
                        else:
                            tagNames.append('Unknown')
                else:
                    s = score.numpy()[indexes]
                    #print(f'scoure: {s}')
                    if s > score_threshold:
                        tagNames.append(name[person_ids[indexes]])
                    else:
                        tagNames.append('Unknown')

                #tracked_bboxes = np.hstack((tracked_bboxes, tagNames))
                draw_temp_boxes(frame, tracked_bboxes, tagNames)

            toc = time.time()
            print(f'FPS: {1 / (toc - tic)}')
            cv2.imshow('Test', frame)

            key = cv2.waitKey(1)
            if key == 'q':
                break

    cv2.destroyAllWindows()

