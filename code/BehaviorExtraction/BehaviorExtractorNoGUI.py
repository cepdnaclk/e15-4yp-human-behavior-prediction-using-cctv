from __future__ import division
import cv2
import time
import datetime
from torch.autograd import Variable
########################################################################################################################
from BehaviorExtraction.yolo.models import *
from BehaviorExtraction.yolo.utils import *
from BehaviorExtraction.yolo.datasets import *

YOLO_MODEL_CFG = "ModelFiles/yolo/yolov3.cfg"
YOLO_MODEL_PATH = "ModelFiles/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416
border = 20
########################################################################################################################
from BehaviorExtraction.reid.extractor import HumanReIdentifier

REID_MODEL_PATH = 'ModelFiles/reid/reid_model.pth'
REID_FEAT_MAT = f'ImageDatabase/Human/{datetime.date.today()}/today.mat'

########################################################################################################################
from BehaviorExtraction.openpose import util
from BehaviorExtraction.openpose.body import Body
from BehaviorExtraction.action.multi_classifier import MultiPersonClassifier

OPENPOSE_MODEL_PATH = 'ModelFiles/openpose/openpose_model.pth'
ACTION_MODEL_PATH = 'ModelFiles/action/model.pickle'
ACTION_CLASSES = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
ACTION_BUFFER_SIZE = 5  # Action recognition: number of frames used to extract features.

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

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    torch.backends.cudnn.benchmark = True

    ###################################################### YOLO ########################################################
    yolo = Darknet(YOLO_MODEL_CFG, img_size=image_size).to(device)
    yolo.load_darknet_weights(YOLO_MODEL_PATH)

    yolo.cuda()
    yolo.eval()  # Set in evaluation mode
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    ###################################################### ReID ########################################################
    REID_FEAT_MAT = f'ImageDatabase/Human/2021-04-15/today.mat'
    reIdentifier = HumanReIdentifier(REID_MODEL_PATH, REID_FEAT_MAT)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ############################################### ACTION RECOGNITION #################################################
    body_estimator = Body(OPENPOSE_MODEL_PATH, device)
    action_classifier = MultiPersonClassifier(ACTION_MODEL_PATH, ACTION_CLASSES, ACTION_BUFFER_SIZE)

    video = cv2.VideoCapture('Demo/CCTV_1.mp4')
    if video.isOpened():
        check, currFrame = video.read()
        orig_h, orig_w = currFrame.shape[:2]

        # orig_w = int(orig_w * 0.5)
        # orig_h = int(orig_h * 0.5)

        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (image_size / max(orig_h, orig_w))
        pad_y = max(orig_w - orig_h, 0) * (image_size / max(orig_h, orig_w))

        # Image height and width after padding is removed
        unpad_h = image_size - pad_y
        unpad_w = image_size - pad_x

    while video.isOpened():
        check, currFrame = video.read()
        tic = time.time()
        # currFrame = cv2.resize(currFrame, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(currFrame)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416).unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = yolo(imgTensor)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        human_crops = []
        human_reids = []
        human_boxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if int(cls_pred) == 0:  # Only Detect Humans
                # Rescale bounding boxes to dimension of original image
                x1 = int(abs(((x1 - pad_x // 2) / unpad_w) * orig_w))
                y1 = int(abs(((y1 - pad_y // 2) / unpad_h) * orig_h))
                x2 = int(abs(((x2 - pad_x // 2) / unpad_w) * orig_w))
                y2 = int(abs(((y2 - pad_y // 2) / unpad_h) * orig_h))

                x1 = x1 - border if x1 - border > 0 else x1
                y1 = y1 - border if y1 - border > 0 else y1
                x2 = x2 + border if x2 + border < orig_w else x2
                y2 = y2 + border if y2 + border < orig_h else y2

                human = currFrame[y1:y2, x1:x2]
                human_reids.append(data_transform(human))
                human_crops.append(human)
                human_boxes.append([x1, y1, x2, y2])

        if human_crops:
            tags = reIdentifier.getHumanTags(human_reids)
            dict_tag2skeleton = {}
            for tag, human_crop, human_box in zip(tags, human_crops, human_boxes):
                if tag > -1:
                    x1, y1, x2, y2 = human_box
                    frameSmall = cv2.resize(human_crop, (int(0.5 * (x2-x1)), int(0.5 * (y2-y1))), interpolation=cv2.INTER_AREA)
                    torch.cuda.empty_cache()

                    candidate, subset = body_estimator(frameSmall)
                    util.draw_bodypose(currFrame, candidate, subset, x1, y1)
                    dict_tag2skeleton[tag] = convertSkeleton(candidate, subset)

            id2action = action_classifier.classify(dict_tag2skeleton)
            draw_temp_boxes(currFrame, human_boxes, tags, id2action)

        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_RGB2BGR)
        toc = time.time()
        print(f'FPS:{round(1 / (toc - tic), 2)}')
        cv2.imshow('Camera', currFrame)
        torch.cuda.empty_cache()
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cv2.destroyAllWindows()