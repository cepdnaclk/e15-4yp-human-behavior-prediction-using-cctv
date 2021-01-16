import random
import cv2
import colorsys
import numpy as np
import math
from yolo.configs import *

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

def draw_track_boxes(image, tracked_bboxes):
    for id, x1, y1, x2, y2 in tracked_bboxes:

        cv2.rectangle(image, (x1, y1), (x2, y2), colors[id] , 2)

        text = f'Tag{id}: Walk'
        (text_width, text_height) , baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_think)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), colors[id], thickness=cv2.FILLED)
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, [0, 0, 0], font_think, lineType=cv2.LINE_AA)

def draw_skel_boxes(image, tracked_bboxes):
    for x1, y1, x2, y2 in tracked_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), [255,0,0] , 2)



#Gonna Remove
def draw_boxes(image, tracked_bboxes):
    '''
    image_h, image_w, _ = image.shape

    num_ids = 10
    hsv_tuples = [(1.0 * x / num_ids, 1., 1.) for x in range(num_ids)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    '''

    cv2.polylines(image, [pts], True, (0, 0, 255), 2)

    for bbox in tracked_bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])

        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[int(bbox[4])], bbox_thick * 2)

        # get text label
        score_str = " {:.2f}".format(score) if show_confidence else ""

        if tracking: score_str = " " + str(score)

        try:
            label = "{}".format(NUM_CLASS[class_ind]) + score_str
        except KeyError:
            print("You received KeyError, this might be that you are trying to use yolo original weights")
            print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), colors[int(bbox[4])], thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)


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

# Gonna Remove
def draw_human_skeleton(img_disp, humans, dict_id2skeleton, dict_id2label, scale_h,  skeleton_detector):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Resize to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp, dsize=(desired_cols, img_disp_desired_rows))

    # Draw all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            skeleton[1::2] = skeleton[1::2] / scale_h
            draw_action_result(img_disp, id, skeleton, label)
    return img_disp

# Gonna Remove
def draw_action_result(img_display, id, skeleton, str_action_label):
    font = cv2.FONT_HERSHEY_SIMPLEX

    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    i = 0
    NaN = 0

    while i < len(skeleton):
        if not(skeleton[i] == NaN or skeleton[i+1] == NaN):
            minx = min(minx, skeleton[i])
            maxx = max(maxx, skeleton[i])
            miny = min(miny, skeleton[i+1])
            #maxy = max(maxy, skeleton[i+1])
        i += 2

    minx = int(minx * img_display.shape[1])
    miny = int(miny * img_display.shape[0])
    maxx = int(maxx * img_display.shape[1])
    maxy = int(maxy * img_display.shape[0])

    # Draw text at left corner
    box_scale = max(0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5)))
    fontsize = 1.2 * box_scale
    linewidth = int(math.ceil(3 * box_scale))

    TEST_COL = int(minx + 5 * box_scale)
    TEST_ROW = int(miny - 10 * box_scale)

    img_display = cv2.putText(img_display, "P"+str(id % 10)+": "+str_action_label, (TEST_COL, TEST_ROW), font, fontsize, (0, 0, 255), linewidth, cv2.LINE_AA)