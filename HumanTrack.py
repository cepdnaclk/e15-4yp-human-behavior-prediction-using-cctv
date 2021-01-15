import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, draw_path, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import pickle
import struct
import socket

################################################## Settings ############################################################

HOST = '127.0.0.1'
PORT = 8083
CONNECTION_ENABLE = True
SRC_VIDEO_SAMPLE_INTERVAL = 1
SRC_VIDEO_PATH = 0 #"assets/TwoHuman.mp4"
SRC_FLOOR_PLAN = "assets/floor_plan.png"
SRC_DEEPSORT_MODEL_PATH = 'model_data/mars-small128/mars-small128.pb'

########################################################################################################################

def Object_tracking(input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only=[]):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize yolo detection object
    yolo = Load_Yolo_model()

    # initialize deep sort object
    encoder = gdet.create_box_encoder(SRC_DEEPSORT_MODEL_PATH, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    vid = cv2.VideoCapture(SRC_VIDEO_PATH)

    if CONNECTION_ENABLE:
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((HOST, PORT))

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    floor_plan = cv2.imread(SRC_FLOOR_PLAN)

    while True:
        check, frame = vid.read()
        key = cv2.waitKey(1)
        if not check or key == 'q':
            print("Video_Stopped")
            break

        tic = time.time()
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = yolo.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
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
            tracked_bboxes.append([track.track_id]+[round(i, 0) for i in track.to_tlbr().tolist()])

        data_bag = {}
        data_bag['frame'] = frame
        data_bag['boxes'] = tracked_bboxes

        #frame = draw_bbox(frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        #floor_plan = draw_path(floor_plan, tracked_bboxes, human_paths)
        print(tracked_bboxes)
        if CONNECTION_ENABLE:
            data = pickle.dumps(data_bag)
            clientsocket.sendall(struct.pack("L", len(data)) + data)

        toc = time.time()
        print(f'FPS: {1 / (toc - tic)}')


Object_tracking(input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255, 0, 0),Track_only=["person"])


