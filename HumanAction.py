import numpy as np
import cv2
import pickle
import struct
import socket
import argparse

import utils.lib_plot as lib_plot
import utils.lib_commons as lib_commons
from utils.lib_openpose import SkeletonDetector
from utils.lib_tracker import Tracker
from utils.lib_tracker import Tracker
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import all sklearn related libraries

################################################## Settings ############################################################
SRC_MODEL_PATH    = 'model/action_classifier.pickle'
CLASSES           = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
OPENPOSE_MODEL    = 'mobilenet_thin'
OPENPOSE_IMG_SIZE = [656, 368] # 656x368 432x368, 336x288. Bigger is more accurate.
WINDOW_SIZE       = 5          # Action recognition: number of frames used to extract features.

# Display Settings
img_disp_desired_rows = 480

# Network Settings
HOST = '127.0.0.1'
PORT = 8083
CONNECTION_ENABLE = False
########################################################################################################################

class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

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

def draw_result_img(img_disp, humans, dict_id2skeleton, skeleton_detector):
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
            lib_plot.draw_action_result(img_disp, id, skeleton, label)
    return img_disp

########################################################################################################################
if __name__ == "__main__":

    # -----------------------------------------------------------------------------------------------------------------#
    # Network Intialization
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b''
    payload_size = struct.calcsize("L")
    # -----------------------------------------------------------------------------------------------------------------#
    # -- Detector, tracker, classifier
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()
    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    # -- Read images and process
    ith_img = -1
    while True:
        #--------------------------------------------------------------------------------------------------------------#
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
        tracked_bboxes = data_bag['boxes']
        print(tracked_bboxes)
        # --------------------------------------------------------------------------------------------------------------#

        # -- Detect skeletons
        humans = skeleton_detector.detect(current_image)
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        skeletons = remove_skeletons_with_few_joints(skeletons)

        # -- Track people
        dict_id2skeleton = multiperson_tracker.track(skeletons)  # int id -> np.array() skeleton

        # -- Recognize action of each person
        dict_id2label = None
        if len(dict_id2skeleton):
            dict_id2label = multiperson_classifier.classify(dict_id2skeleton)

        # -- Draw
        current_image = draw_result_img(current_image, humans, dict_id2skeleton, skeleton_detector)

        cv2.imshow('Skeleton', current_image)
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cv2.destroyAllWindows()

