import numpy as np
import cv2
import pickle
import struct
import socket
import PySimpleGUI as sg
import datetime
import pymongo

import utils.lib_commons as lib_commons
from utils.lib_openpose import SkeletonDetector
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import all sklearn related libraries
from utils.lib_draw import draw_track_boxes, draw_skel_boxes, draw_human_path, draw_human_skeleton
########################################################################################################################

from Configurations import *
from tf_pose.common import CocoPart
################################################## Settings ############################################################
ACTION_MODEL_PATH    = 'model_data/action_classifier/model.pickle'
ACTION_CLASSES           = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
ACTION_BUFFER_SIZE       = 5          # Action recognition: number of frames used to extract features.

OPENPOSE_MODEL_PATH    = 'model_data/mobilenet_thin/graph_opt.pb'
OPENPOSE_IMG_SIZE = [656, 368] # 656x368 432x368, 336x288. Bigger is more accurate.

SRC_FLOOR_PLAN = "assets/floor_plan.png"

DB_NAME = "ProjectDB"
USERNAME = "Risith"
PASSWORD = "Risith#1234"
########################################################################################################################

class MultiPersonClassifier(object):

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(model_path, classes, ACTION_BUFFER_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]

def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 4 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons

def track_skeleton(humans, tracked_boxes):
    tracked_skel = {}
    scale_h = 1.0 * frame_h / frame_w
    for human in humans:
        # draw point
        xs, ys = [], []
        skeleton = [0] * (18 * 2)
        for i in range(CocoPart.Background.value): #Value 18
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            idx = body_part.part_idx
            skeleton[2 * idx] = body_part.x
            skeleton[2 * idx + 1] = body_part.y * scale_h

            if i not in [3, 4, 6, 7]: # Ignore Hands
                xs.append(int(body_part.x * frame_w + 0.5))
                ys.append(int(body_part.y * frame_h + 0.5))

        in_box = [min(xs), min(ys), max(xs), max(ys)]
        in_box_area = abs(in_box[2] - in_box[0]) * abs(in_box[1] - in_box[3])
        if not in_box_area:
            continue

        for t_box in tracked_boxes:
            x_dist = (min(t_box[3], in_box[2]) - max(t_box[1], in_box[0]))
            y_dist = (min(t_box[4], in_box[3]) - max(t_box[2], in_box[1]))

            overlap_area = 0
            if x_dist > 0 and y_dist > 0:
                overlap_area = x_dist * y_dist

            if overlap_area / in_box_area >= 0.9:
                tracked_skel[t_box[0]] = skeleton
                break

    return tracked_skel

def get_final_image(video, floor, size=[600, 1066]): #(height, width) format
    video_size = video.shape[:2]
    ratio_h = size[0] / video_size[0]
    ratio_w = size[1] / video_size[1]

    #new_video_size = tuple([int(i * ratio_w) for i in video_size])
    #print(new_video_size)
    new_video_size = tuple([int(i * ratio_h) for i in video_size])
    #print(new_video_size)

    #new_video_size = tuple([int(i * ratio) for i in video_size])

    resized_video = cv2.resize(video, (new_video_size[1], new_video_size[0]))

    delta_w = size[1] - new_video_size[1]
    delta_h = size[0] - new_video_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    #print([top, bottom, left, right])
    padded_video = cv2.copyMakeBorder(resized_video, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    floor_size = floor.shape[:2]
    new_floor_size = tuple([int(x * float(size[0]) / floor_size[0]) for x in floor_size])
    new_floor = cv2.resize(floor, (new_floor_size[1], new_floor_size[0]))
    return np.concatenate((padded_video, new_floor), axis=1)

########################################################################################################################
if __name__ == "__main__":
    # ----------------------------------------------- MongoDB ---------------------------------------------------------#
    client = pymongo.MongoClient(f'mongodb+srv://{USERNAME}:{PASSWORD}@projectcluster.unskd.mongodb.net/{DB_NAME}?retryWrites=true&w=majority')
    db = client[DB_NAME]
    counter = db.counters.find_one({"_id": "recordId"})
    next_id = counter["nextId"]

    vid = cv2.VideoCapture(0)

    # --------------------------------------------- Open Pose ---------------------------------------------------------#
    # Initialize Pose Detector, Action Classifier
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL_PATH, OPENPOSE_IMG_SIZE)
    action_classifier = MultiPersonClassifier(ACTION_MODEL_PATH, ACTION_CLASSES)

    # -- Read images and process
    floor_plan = cv2.imread(SRC_FLOOR_PLAN)

    # ------------------------------------------------ GUI ------------------------------------------------------------#
    btn_style = {'size': (5, 1), 'font': ('Franklin Gothic Book', 24), 'button_color': ("black", "#F8F8F8")}
    chekc_style = {'size': (25, 1), 'font': ('Franklin Gothic Book', 16)}

    check_box_column = [
        [sg.Text("Choose Following Options", font=('Franklin Gothic Book', 16))],
        [sg.Checkbox(': Draw Boxes', **chekc_style, default=True, key="Check_1")],
        [sg.Checkbox(': Draw Skeleton', **chekc_style, default=True, key="Check_2")],
        [sg.Checkbox(': Draw Paths', **chekc_style, default=True, key="Check_3")],
    ]

    layout = [
        [sg.Image(filename='', key='video')],
        [sg.Multiline(size=(105, 30), font='courier 12', background_color='black', text_color='white', key='mline'), sg.VSeperator(), sg.Column(check_box_column)],
    ]

    # create the window and show it without the plot
    window = sg.Window('Human Behaviour Prediction', layout, size=(1565, 800), no_titlebar=False)

    # locate the elements we'll be updating. Does the search only 1 time
    video = window['video']
    multiline = window['mline']
    # -----------------------------------------------------------------------------------------------------------------#

    while True:
        event, values = window.read(timeout=0)
        if event is None:
            break

        #--------------------------------------------------------------------------------------------------------------#
        if CONNECTION_ENABLE:
            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_msg_size = data[:payload_size]

            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            data_bag = pickle.loads(frame_data)
            current_frame = data_bag['frame']
            tracked_boxes = data_bag['boxes']
        else:
            check, current_frame = vid.read(0)
        # --------------------------------------------------------------------------------------------------------------#

        frame_h, frame_w = current_frame.shape[:2]
        scale_h = 1.0 * frame_h / frame_w

        # -- Detect Skeletons
        humans = skeleton_detector.detect(current_frame)

        # -- Track Skeletons
        dict_id2skeleton = track_skeleton(humans, tracked_boxes)
        #skeletons = remove_skeletons_with_few_joints(skeletons)

        # -- Recognize Action
        id2label = action_classifier.classify(dict_id2skeleton)
        #print(id2label)

        # -- Draw
        #if CONNECTION_ENABLE:
        if values["Check_1"] == True:
            draw_track_boxes(current_frame, tracked_boxes, id2label)

        if values["Check_2"] == True:
            draw_human_skeleton(current_frame, humans)

        locations = {}
        if values["Check_3"] == True:
            floor_plan, locations = draw_human_path(floor_plan, tracked_boxes)

        for id, x1, y1, x2, y2 in tracked_boxes:
            action = 'None'
            if id in id2label:
                if id2label[id] != '':
                    action = f'{id2label[id]}'

            loca = 'None'
            if id in locations:
                loca = f'{locations[id]}'

            time= datetime.datetime.now()

            record = {"_id":next_id, "Date":time.strftime("%Y-%m-%d"), "Time":time.strftime("%H:%M:%S"),"Tag":id, "Location":loca, "Pose":action}
            multiline.update(record)
            next_id = next_id + 1
            db.dataRecords.insert_one(record)
            db.counters.update_one({"_id": "recordId"}, {"$inc": {"nextId": 1}})
            print(record)

        imgbytes = cv2.imencode('.png', get_final_image(current_frame, floor_plan))[1].tobytes()
        video.update(data=imgbytes)