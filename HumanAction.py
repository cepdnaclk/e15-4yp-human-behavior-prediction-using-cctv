import numpy as np
import cv2
import pickle
import struct
import socket
import argparse

import utils.lib_commons as lib_commons
from utils.lib_openpose import SkeletonDetector
from utils.lib_tracker import Tracker
from utils.lib_tracker import Tracker
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import all sklearn related libraries
from utils.lib_draw import draw_track_boxes, draw_skel_boxes, draw_human_path, draw_human_skeleton
########################################################################################################################

from yolo.configs import *
################################################## Settings ############################################################
CLASSIFIER_MODEL_PATH    = 'model_data/action_classifier/model.pickle'
ACTION_CLASSES           = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
OPENPOSE_MODEL_PATH    = 'model_data/mobilenet_thin/graph_opt.pb'
OPENPOSE_IMG_SIZE = [656, 368] # 656x368 432x368, 336x288. Bigger is more accurate.
WINDOW_SIZE       = 5          # Action recognition: number of frames used to extract features.

SRC_FLOOR_PLAN = "assets/floor_plan.png"
########################################################################################################################

class MultiPersonClassifier(object):

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(model_path, classes, WINDOW_SIZE, human_id)

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

########################################################################################################################
if __name__ == "__main__":

    # -----------------------------------------------------------------------------------------------------------------#
    # Network Intialization
    if CONNECTION_ENABLE:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')
        s.bind((HOST, PORT))
        print('Socket bind complete')
        s.listen(10)
        print('Socket now listening')

        conn, addr = s.accept()
        data = b''
        payload_size = struct.calcsize("L")
    else:
        vid = cv2.VideoCapture(SRC_VIDEO_PATH)
    # -----------------------------------------------------------------------------------------------------------------#

    # -- Detector, tracker, classifier
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL_PATH, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()
    multiperson_classifier = MultiPersonClassifier(CLASSIFIER_MODEL_PATH, ACTION_CLASSES)

    # -- Read images and process
    floor_plan = cv2.imread(SRC_FLOOR_PLAN)

    while True:
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
            current_image = data_bag['frame']
            tracked_boxes = data_bag['boxes']
        else:
            check, current_image = vid.read()
        # --------------------------------------------------------------------------------------------------------------#

        # -- Detect skeletons
        humans = skeleton_detector.detect(current_image)
        inner_boxes = skeleton_detector.draw(current_image, humans)

        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        skeletons = remove_skeletons_with_few_joints(skeletons)

        # -- Track people
        dict_id2skeleton = multiperson_tracker.track(skeletons)  # int id -> np.array() skeleton

        # -- Recognize action of each person
        dict_id2label = None
        if len(dict_id2skeleton):
            dict_id2label = multiperson_classifier.classify(dict_id2skeleton)

        # -- Draw
        if CONNECTION_ENABLE:
            draw_track_boxes(current_image, tracked_boxes)
            #floor_plan = draw_human_path(floor_plan, tracked_bboxes)



        draw_skel_boxes(current_image, boxes)
        #current_image = draw_human_skeleton(current_image, humans, dict_id2skeleton, dict_id2label, scale_h, skeleton_detector)

        cv2.imshow('CCTV Stream', current_image)
        cv2.imshow('FloorPlan', floor_plan)
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cv2.destroyAllWindows()

