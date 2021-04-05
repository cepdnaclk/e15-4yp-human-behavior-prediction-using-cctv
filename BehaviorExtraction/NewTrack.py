from __future__ import division
import cv2
########################################################################################################################
import torch.backends.cudnn as cudnn
from BehaviorExtraction.reid.model import ft_net
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import scipy.io

# Options
name = 'ft_ResNet50'
data_path = 'database/2021-02-05'
model_path = 'model_data/ft_ResNet50_ReID/net_last.pth'
mat_path = 'database/2021-02-05/features.mat'
stride = 2
batchsize = 256
nclasses = 751
score_threshold = 0.5
gpu_id = 0

########################################################################################################################
from BehaviorExtraction.yolo2.models import *
from BehaviorExtraction.yolo2.utils import *
from BehaviorExtraction.yolo2.datasets import *

model_def = "model_data_new/yolo/yolov3.cfg"
weights_path = "model_data_new/yolo/yolov3.weights"

conf_thres = 0.8
nms_thres = 0.4
image_size = 416
dist_thres = 0.8
face_thres = 0.9
save_time_thres = 20 #In seconds

def draw_temp_boxes(image, tracked_bboxes, names):
    for box, name in zip(tracked_bboxes, names):
        x1, y1, x2, y2 = box
        cv2.rectangle(currFrame, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), [255, 0, 0] , 2)

        (text_width, text_height) , baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), [255, 0, 0] , thickness=cv2.FILLED)
        cv2.putText(image, name, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [255, 255, 255], 1, lineType=cv2.LINE_AA)

def extract_feature(model, img):
    feature = torch.FloatTensor()
    n, c, h, w = img.size()
    ff = torch.FloatTensor(n, 512).zero_().cuda()
    for i in range(2):
        if (i == 1):  # Flip the image
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
            img = img.index_select(3, inv_idx)

        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    feature = torch.cat((feature, ff.data.cpu()), 0)
    return feature

if __name__ == "__main__":
    ###################################################### YOLO ########################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    yolo = Darknet(model_def, img_size=image_size).to(device)
    yolo.load_darknet_weights(weights_path)

    yolo.cuda()
    yolo.eval() # Set in evaluation mode
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    ###################################################### ReID ########################################################
    # Set gpu ids
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    result = scipy.io.loadmat(mat_path)
    person_feature = torch.FloatTensor(result['person_feature'])
    person_ids = result['person_ids'][0]

    model = ft_net(nclasses, stride=stride)
    model.load_state_dict(torch.load(model_path))
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    name = ['Aunty', 'Risith', 'Asith', 'Madaya', 'YellowMan', 'BlueMan']
    ####################################################################################################################

    video = cv2.VideoCapture(0) #"./database/HumanVideo.mp4")
    if video.isOpened():
        check, currFrame = video.read()
        orig_h, orig_w = currFrame.shape[:2]

        #orig_w = int(orig_w * 0.5)
        #orig_h = int(orig_h * 0.5)

        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (image_size / max(orig_h, orig_w))
        pad_y = max(orig_w - orig_h, 0) * (image_size / max(orig_h, orig_w))

        # Image height and width after padding is removed
        unpad_h = image_size - pad_y
        unpad_w = image_size - pad_x


    while video.isOpened():
        check, currFrame = video.read()
        #currFrame = cv2.resize(currFrame, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(currFrame)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416).unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = yolo(imgTensor)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        human_crops = []
        human_boxes = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if int(cls_pred) == 0: #Only Detect Humans
                # Rescale bounding boxes to dimension of original image
                x1 = int(abs(((x1 - pad_x // 2) / unpad_w) * orig_w))
                y1 = int(abs(((y1 - pad_y // 2) / unpad_h) * orig_h))
                x2 = int(abs(((x2 - pad_x // 2) / unpad_w) * orig_w))
                y2 = int(abs(((y2 - pad_y // 2) / unpad_h) * orig_h))

                human = currFrame[y1:y2, x1:x2]
                #image = cv2.cvtColor(human, cv2.COLOR_BGR2RGB)
                human_crops.append(data_transform(human))
                human_boxes.append([x1, y1, x2, y2])

        if human_crops:
            stacked_tensor = torch.stack(human_crops)
            with torch.no_grad():
                feature = extract_feature(model, stacked_tensor)

            query = torch.transpose(feature, 0, 1)
            score = torch.mm(person_feature, query)
            score = score.squeeze(1).cpu()

            tagNames = []
            indexes = np.argmax(score.numpy(), axis=0)

            '''
            if type(indexes) is np.int64:
                indexes = [indexes]

            print(indexes)
            
            for i in indexes:
                tagNames.append(name[person_ids[i]])
            '''

            if type(indexes) is not np.int64:
                scores = np.diag(score.numpy()[indexes])
                print(scores)
                for i, s in zip(indexes, scores):
                    if s > score_threshold:
                        tagNames.append(name[person_ids[i]])
                    else:
                        tagNames.append('Unknown')
            else:
                s = score.numpy()[indexes]
                print(f'scoure: {s}')
                if s > score_threshold:
                    tagNames.append(name[person_ids[indexes]])
                else:
                    tagNames.append('Unknown')

            print(tagNames)
            draw_temp_boxes(currFrame, human_boxes, tagNames)

        #cv2.imshow('Humans', human)
        cv2.imshow('Camera', currFrame)

        torch.cuda.empty_cache()
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cv2.destroyAllWindows()