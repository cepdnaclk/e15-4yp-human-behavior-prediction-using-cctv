import time
import torch
import cv2
from PIL import Image

# Import modules defined in the 'utilities' folder and models defined in the homonym file
from utilities import utils
import models

################## MODEL INITIALIZATION ##################

# Attach important file paths Define paths to the YOLOv3 trained model
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'

# Define some parameters which drives the model
img_size = 416
conf_thres = 0.8   # 0.6
nms_thres = 0.4    # 0.4

# Load YOLOv3 object detection model
model = models.Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()
classes = utils.load_classes(class_path)
############################################################

# If CUDA is available store the model in the GPU
if torch.cuda.is_available():
    model.cuda()
    print("CUDA Enabled")
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Pack all these information in a single list
model_info = [model, img_size, conf_thres, nms_thres, Tensor]

####################################################################
# Perform object detection in the image and 
# measure the time needed to complete the task

print("[INFO]: Start Video Capture")
video = cv2.VideoCapture(0) #'../Assets/Videos/walk9_slow.mp4')

check, frame = video.read()
if check:
    print(frame.shape)
    frame_H = frame.shape[0]
    frame_W = frame.shape[1]
    scale = frame_W/img_size
    ysub = int((img_size - img_size*frame_H/frame_W) / 2)
else:
    print("Video Error" )
    exit()

while True:
    check, frame = video.read()
    if not check:
        print("Video Stopped")
        break

    prev_time = time.time()
    #################################################################
    pil_img = Image.fromarray(frame)

    detections = utils.detect_image(pil_img, model_info)

    # Draw over the image the boxes (if any) obtained from the 'detect_image' function
    if detections is not None:
        # For each detection, draw the corresponding bounding
        # box and write over it the detected class of the object
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            y1 = (y1 - ysub) * scale
            y2 = (y2 - ysub) * scale
            x1 = x1 * scale
            x2 = x2 * scale
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.rectangle(frame, (x1, y1), (x1 + int((x2-x1)/2), y1+18), (255, 0, 0), -1)
            cv2.circle(frame, (x1 + (x2-x1)/2, y2), 10, (0, 255, 0), -1)
            cv2.putText(frame, classes[int(cls_pred)], (x1, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #print(f'Elapced Time(S): {time.time() - prev_time}')
    fps = round(1 / (time.time() - prev_time), 2)
    cv2.putText(frame, f'FPS:{fps}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    #################################################################
    cv2.imshow("Human Bounding Box", frame)
    #plt.show()

    key = cv2.waitKey(10)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()