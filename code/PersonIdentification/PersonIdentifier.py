from __future__ import division

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time

from PersonIdentification.yolo.models import *
from PersonIdentification.yolo.utils import *
from PersonIdentification.yolo.datasets import *

from torch.autograd import Variable

from pathlib import Path
import datetime

########################################################################################################################
data_path = "ImageDatabase/Faces/faces.pt"
YOLO_MODEL_CFG = "ModelFiles/yolo/yolov3.cfg"
YOLO_MODEL_PATH = "ModelFiles/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416

dist_thres = 0.8
face_thres = 0.9
save_time_thres = 5 #In seconds

########################################################################################################################
def draw_temp_boxes(image, tracked_bboxes, names):
    for box, name in zip(tracked_bboxes, names):
        x1, y1, x2, y2 = box
        #cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0], 2)

        (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [255, 0, 0], thickness=cv2.FILLED)
        cv2.putText(image, name, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [255, 255, 255], 1,lineType=cv2.LINE_AA)


########################################################################################################################
class PersonIdentifier():
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

        ########################################################################################################################
        # Initializing mtcnn for face detection
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # Initializing resnet for face img to embeding conversion
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        saved_data = torch.load(data_path)  # loading faces.pt file
        self.embedding_list = saved_data[0]  # getting embedding data
        self.tag_list = saved_data[1]  # getting list of names

        self.human_save_path = f'ImageDatabase/Human/{datetime.date.today()}/'
        self.human_save_count = dict.fromkeys(set(self.tag_list), 0)
        self.human_save_time = dict.fromkeys(set(self.tag_list), None)

        # Get current num of photos in todays folder in each tag
        for tag in self.tag_list:
            self.human_save_count[tag] = len(glob.glob1(self.human_save_path + f'{tag}', "*.jpg"))

    def detect(self, currFrame):
        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(currFrame)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416).unsqueeze(0)
        imgTensor = Variable(imgTensor.type(self.Tensor))

        with torch.no_grad():
            detections = self.yolo(imgTensor)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        count = True
        humanFrame = np.zeros([241, 100, 3])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if int(cls_pred) == 0: #Only Detect Humans
                # Rescale bounding boxes to dimension of original image
                x1 = int(abs(((x1 - self.pad_x // 2) / self.unpad_w) * self.orig_w))
                y1 = int(abs(((y1 - self.pad_y // 2) / self.unpad_h) * self.orig_h))
                x2 = int(abs(((x2 - self.pad_x // 2) / self.unpad_w) * self.orig_w))
                y2 = int(abs(((y2 - self.pad_y // 2) / self.unpad_h) * self.orig_h))

                human = currFrame[y1:y2, x1:x2]
                humanPIL = Image.fromarray(human)
                face, prob = self.mtcnn(humanPIL, save_path='face.png', return_prob=True)
                if face is not None:
                    probArgMax = prob.argmax()
                    if prob[probArgMax] > face_thres: # Check the probability to be a human face
                        p = time.time()
                        emb = self.resnet(face[probArgMax].unsqueeze(0)).detach()
                        q = time.time()
                        print(f'RESNET Time: {round(q-p,3)} ({round(1/(q-p),2)})')
                        dist_list = []  # list of matched distances, minimum distance is used to identify the person
                        for idx, emb_db in enumerate(self.embedding_list):
                            dist = torch.dist(emb, emb_db).item()
                            dist_list.append(dist)

                        minDist = min(dist_list)
                        if minDist < dist_thres:
                            idx_min = dist_list.index(minDist)
                            tag = self.tag_list[idx_min]
                            currTime = datetime.datetime.now()
                            if self.human_save_time[tag] is None or save_time_thres < (currTime - self.human_save_time[tag]).total_seconds():
                                save_path = f'{self.human_save_path}{tag}'
                                Path(save_path).mkdir(parents=True, exist_ok=True)
                                human = cv2.cvtColor(human, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(save_path, f'{tag}_{self.human_save_count[tag]}.png'), human)
                                self.human_save_count[tag] = self.human_save_count[tag] + 1
                                self.human_save_time[tag] = currTime
                        else:
                            tag = 'Unknown'

                        human = cv2.resize(human, (100, 241), interpolation=cv2.INTER_AREA)
                        human = cv2.cvtColor(human, cv2.COLOR_RGB2BGR)
                        cv2.putText(human, tag , (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if count:
                            humanFrame = human
                            count = False
                        else:
                            humanFrame = np.concatenate((humanFrame, human), axis=1)

        torch.cuda.empty_cache()
        return  currFrame, humanFrame