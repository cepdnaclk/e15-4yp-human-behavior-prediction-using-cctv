import random
import cv2
import colorsys
import numpy as np
import math
from yolo.configs import *
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

            #if i not in [3, 4, 6, 7]:
                #xs.append(center_x)
                #ys.append(center_y)

            #bboxes.append([min(xs), min(ys), max(xs), max(ys)])
        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            cv2.line(frame, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

def draw_human_path(floor_plan, bboxes):
    num_ids = 10
    hsv_tuples = [(1.0 * x / num_ids, 1., 1.) for x in range(num_ids)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i, bbox in enumerate(bboxes):
        cur_point_conv = cv2.perspectiveTransform(bbox[6], h_mat)
        if cur_point_conv is not None:
            cur_point = cur_point_conv[0][0]
            track_id = int(bbox[4])
            if pre_point is not None:
                # pre_point_conv = cv2.perspectiveTransform(pre_point, h_mat)
                # pre_point = pre_point_conv[0][0]
                # print(f'{track_id}----> ({pre_point[0]} :: {pre_point[1]}) ({cur_point[0]} :: {cur_point[1]})')
                cv2.circle(floor_plan, (cur_point[0], cur_point[1]), radius=0, color=(0, 0, 255), thickness=-1)
                # cv2.line(floor_plan, (pre_point[0], pre_point[1]), (cur_point[0], cur_point[1]), colors[track_id], 5)

    return floor_plan
