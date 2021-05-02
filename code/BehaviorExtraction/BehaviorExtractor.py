from __future__ import division
import cv2
import time
import datetime
from torch.autograd import Variable

from BehaviorExtraction.yolo.models import *
from BehaviorExtraction.yolo.utils import *
from BehaviorExtraction.yolo.datasets import *

from BehaviorExtraction.reid.extractor import HumanReIdentifier

from BehaviorExtraction.openpose import util
from BehaviorExtraction.openpose.body import Body
from BehaviorExtraction.action.multi_classifier import MultiPersonClassifier
########################################################################################################################

YOLO_MODEL_CFG = "ModelFiles/yolo/yolov3.cfg"
YOLO_MODEL_PATH = "ModelFiles/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416
border = 20

REID_MODEL_PATH = 'ModelFiles/reid/reid_model.pth'
REID_FEAT_MAT = f'ImageDatabase/Human/{datetime.date.today()}/today.mat'

OPENPOSE_MODEL_PATH = 'ModelFiles/openpose/openpose_model.pth'
ACTION_MODEL_PATH = 'ModelFiles/action/model.pickle'
ACTION_CLASSES = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
ACTION_BUFFER_SIZE = 5  # Action recognition: number of frames used to extract features.

pts_src = np.array([[470, 412], [1251, 557], [1155, 715], [194, 720], [24, 520]])
pts_dst = np.array([[426, 9], [426, 481], [221, 475], [60, 381], [128, 9]])
h_mat, status = cv2.findHomography(pts_src, pts_dst)
pts = np.array([[140, 678], [336, 259], [385, 147], [736, 80], [916, 35], [1113, 571]], np.int32)
pts = pts.reshape((-1, 1, 2))

colors = [[0, 0, 255], [255, 0, 255], [255, 170, 0], [255, 255, 0]]
########################################################################################################################

def draw_temp_boxes(image, tracked_bboxes, tags, id2action):
    for box, tag in zip(tracked_bboxes, tags):
        if tag == -1:
            text = '******'
        elif tag not in id2action or id2action[tag] == '':
                text = f'Tag{tag} :***'
        else:
            text = f'Tag{tag} :{id2action[tag]}'

        x1, y1, x2, y2 = box
        #cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0], 2)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [0, 0, 255], thickness=cv2.FILLED)
        cv2.putText(image, text, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, [255, 255, 255], 1,lineType=cv2.LINE_AA)


def draw_human_path(floor_plan, tracked_bboxes, tags):
    locations = {}
    for box, tag in zip(tracked_bboxes, tags):
        x1, y1, x2, y2 = box
        real_point = np.array([np.array([[x1 + (x2 - x1) / 2, y2]], dtype='float32')])
        trans_point = cv2.perspectiveTransform(real_point, h_mat)
        if trans_point is not None:
            cur_point = trans_point[0][0]
            locations[tag] = cur_point
            floor_plan = cv2.circle(floor_plan, (cur_point[0], cur_point[1]), radius=5, color= colors[tag], thickness=-1)

    return locations

def convertSkeleton(candidate, subset):
    subset = subset.astype(int).tolist()
    flatSkeleton = [0] * 36

    for subpoint in subset:
        subpoint = [num for num in subpoint[0:18] if num is not -1]
        skeleton = {}
        for point in subpoint:
            skeleton[point] = candidate[point][0:2].tolist()

        for i in range(18):
            if i in skeleton.keys():
                flatSkeleton[2 * i] = skeleton[i][0]
                flatSkeleton[2 * i + 1] = skeleton[i][1]

    return flatSkeleton
########################################################################################################################

class BehaviorExtractor():
    def __init__(self, width, height):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {device}')
        torch.backends.cudnn.benchmark = True
        self.rid = 0
        self.orig_w = int(width)  # int(orig_w * 0.5)
        self.orig_h = int(height)  # int(orig_h * 0.5)

        # The amount of padding that was added
        self.pad_x = max(self.orig_h - self.orig_w, 0) * (image_size / max(self.orig_h, self.orig_w))
        self.pad_y = max(self.orig_w - self.orig_h, 0) * (image_size / max(self.orig_h, self.orig_w))

        # Image height and width after padding is removed
        self.unpad_h = image_size - self.pad_y
        self.unpad_w = image_size - self.pad_x

        ###################################################### YOLO ########################################################
        self.yolo = Darknet(YOLO_MODEL_CFG, img_size=image_size).to(device)
        self.yolo.load_darknet_weights(YOLO_MODEL_PATH)

        self.yolo.cuda()
        self.yolo.eval()  # Set in evaluation mode
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        ###################################################### ReID ########################################################
        self.reIdentifier = HumanReIdentifier(REID_MODEL_PATH, REID_FEAT_MAT)

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        ############################################### ACTION RECOGNITION #################################################
        self.body_estimator = Body(OPENPOSE_MODEL_PATH, device)
        self.action_classifier = MultiPersonClassifier(ACTION_MODEL_PATH, ACTION_CLASSES, ACTION_BUFFER_SIZE)

        ####################################################################################################################
        print("Finished Initialization...")

    def detect(self, currFrame, floor_plan):
        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(currFrame)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416).unsqueeze(0)
        imgTensor = Variable(imgTensor.type(self.Tensor))

        with torch.no_grad():
            detections = self.yolo(imgTensor)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        human_crops = []
        human_reids = []
        human_boxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if int(cls_pred) == 0:  # Only Detect Humans
                # Rescale bounding boxes to dimension of original image
                x1 = int(abs(((x1 - self.pad_x // 2) / self.unpad_w) * self.orig_w))
                y1 = int(abs(((y1 - self.pad_y // 2) / self.unpad_h) * self.orig_h))
                x2 = int(abs(((x2 - self.pad_x // 2) / self.unpad_w) * self.orig_w))
                y2 = int(abs(((y2 - self.pad_y // 2) / self.unpad_h) * self.orig_h))

                x1 = x1 - border if x1 - border > 0 else x1
                y1 = y1 - border if y1 - border > 0 else y1
                x2 = x2 + border if x2 + border < self.orig_w else x2
                y2 = y2 + border if y2 + border < self.orig_h else y2

                human = currFrame[y1:y2, x1:x2]
                human_reids.append(self.data_transform(human))
                human_crops.append(human)
                human_boxes.append([x1, y1, x2, y2])

        records = []
        human_filter_boxes = []
        human_filter_tags = []
        if human_crops:
            tags = self.reIdentifier.getHumanTags(human_reids)
            dict_tag2skeleton = {}
            for tag, human_crop, human_box in zip(tags, human_crops, human_boxes):
                if tag > -1:
                    x1, y1, x2, y2 = human_box
                    human_filter_boxes.append(human_box)
                    human_filter_tags.append(tag)
                    frameSmall = cv2.resize(human_crop, (int(0.5 * (x2 - x1)), int(0.5 * (y2 - y1))), interpolation=cv2.INTER_AREA)
                    torch.cuda.empty_cache()

                    candidate, subset = self.body_estimator(frameSmall)
                    util.draw_bodypose(currFrame, candidate, subset, x1, y1)
                    dict_tag2skeleton[tag] = convertSkeleton(candidate, subset)

                    #########################################################################

            tag2action = self.action_classifier.classify(dict_tag2skeleton)
            draw_temp_boxes(currFrame, human_boxes, tags, tag2action)
            locations = draw_human_path(floor_plan, human_filter_boxes, human_filter_tags)

            for tag in tag2action:
                time = datetime.datetime.now()
                action = 'none' if tag2action[tag] == '' else tag2action[tag]
                action = action.ljust(6)

                record = {"Id": self.rid, "Date": time.strftime("%Y-%m-%d"), "Time": time.strftime("%H:%M:%S"),"Tag": tag,"Pose": action, "X Cor": locations[tag][0], "Y Cor":locations[tag][1]}
                self.rid += 1
                records.append(str(record))

        #currFrame = cv2.polylines(currFrame, [pts_src], True, 255,0,0, 2)
        torch.cuda.empty_cache()
        return currFrame, len(human_boxes), records
