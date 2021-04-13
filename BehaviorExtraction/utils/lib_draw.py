import random
import cv2
import colorsys
import numpy as np
import math
from Configurations import *
from tf_pose.common import CocoPart, CocoColors, CocoPairsRender

pts_src = np.array([[140, 678], [336, 259], [385, 147], [736, 80], [916, 35], [1113, 571]])
pts_dst = np.array([[0, 335], [0, 104], [0, 0], [289, 0], [406, 0], [335, 406]])
h_mat, status = cv2.findHomography(pts_src, pts_dst)
pts = np.array([[140, 678], [336, 259], [385, 147], [736, 80], [916, 35], [1113, 571]], np.int32)
pts = pts.reshape((-1, 1, 2))

hsv_tuples = [(1.0 * x / YOLO_MAX_TRACKING, 1., 1.) for x in range(YOLO_MAX_TRACKING)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

img_disp_desired_rows = 480
box_thick = 2
font_think = 1
font_scale = 0.6

def draw_track_boxes(image, tracked_bboxes, id2label):
    for id, x1, y1, x2, y2 in tracked_bboxes:

        cv2.rectangle(image, (x1, y1), (x2, y2), colors[id] , 2)

        if id in id2label:
            if id2label[id] == '':
                text = f'Tag{id}: Processing...'
            else:
                text = f'Tag{id}: {id2label[id]}'
        else:
            text = f'Tag{id}: Processing...'

        (text_width, text_height) , baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_think)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), colors[id], thickness=cv2.FILLED)
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, [0, 0, 0], font_think, lineType=cv2.LINE_AA)

def draw_skel_boxes(image, tracked_bboxes):
    for x1, y1, x2, y2 in tracked_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), [255,0,0] , 2)

def draw_human_skeleton(frame, humans):
    image_h, image_w = frame.shape[:2]
    for human in humans:
        # draw point
        xs, ys, centers = [], [], {}
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center_x = int(body_part.x * image_w + 0.5)
            center_y = int(body_part.y * image_h + 0.5)
            centers[i] = (center_x, center_y)
            cv2.circle(frame, (center_x, center_y), 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(frame, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 2)

def draw_human_path(floor_plan, tracked_bboxes):
    locations = {}
    for id, x1, y1, x2, y2 in tracked_bboxes:
        real_point = np.array([np.array([[x1+ (x2-x1)/2, y2]], dtype='float32')])
        trans_point = cv2.perspectiveTransform(real_point, h_mat)
        if trans_point is not None:
            cur_point = trans_point[0][0]
            locations[id] = cur_point
            floor_plan = cv2.circle(floor_plan, (cur_point[0], cur_point[1]), radius=2, color= colors[id], thickness=-1)

    return floor_plan, locations
