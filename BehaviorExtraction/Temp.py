from __future__ import division
import cv2
import torch
import numpy as np

########################################################################################################################
from utils.lib_openpose import SkeletonDetector

from openpose.body import Body

OPENPOSE_MODEL_PATH = 'model_data/openpose/openpose_model.pth'
ACTION_MODEL_PATH = 'model_data/action_classifier/model.pickle'
ACTION_CLASSES = np.array(['stand', 'walk', 'walk', 'stand', 'sit', 'walk', 'stand', 'stand', 'stand'])
ACTION_BUFFER_SIZE = 5  # Action recognition: number of frames used to extract features.

TFPOSE_MODEL_PATH    = 'model_data/mobilenet_thin/graph_opt.pb'
TFPOSE_IMG_SIZE = [656, 368] # 656x368 432x368, 336x288. Bigger is more accurate.

# draw the body keypoint and lims
def draw_pytorch_skel(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    px = []
    py = []
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            px.append(x)
            py.append(y)
            cv2.circle(canvas, (int(x), int(y)), 4, [0,255,0], thickness=-1)
            cv2.putText(image, str(i), (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, [0, 0, 0], 1, lineType=cv2.LINE_AA)

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            #cv2.line(canvas, (int(Y[0]), int(X[0])), (int(Y[1]) , int(X[1])), [0,255,0], 2)

    return px,py

def draw_tensor_skel(frame, humans):
    CocoPairs = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
        (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
    ]  # = 19
    CocoPairsRender = CocoPairs[:-2]

    image_h, image_w = frame.shape[:2]
    tx = []
    ty = []
    for human in humans:
        # draw point
        xs, ys, centers = [], [], {}
        for i in range(18):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            tx.append(round(body_part.x * image_w + 0.5, 2))
            ty.append(round(body_part.y * image_h + 0.5, 2))
            center_x = int(body_part.x * image_w + 0.5)
            center_y = int(body_part.y * image_h + 0.5)
            centers[i] = (center_x, center_y)
            cv2.circle(frame, (center_x, center_y), 3, [0,0,255], thickness=3, lineType=8, shift=0)
            cv2.putText(image, str(i), (center_x-10, center_y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, [255, 0, 0], 1, lineType=cv2.LINE_AA)
        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            #cv2.line(frame, centers[pair[0]], centers[pair[1]], [0,0,255], 2)
    return tx, ty



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    torch.backends.cudnn.benchmark = True

    image = cv2.imread('assets/test.png')
    ############################################### ACTION RECOGNITION #################################################
    skeleton_detector = SkeletonDetector(TFPOSE_MODEL_PATH, TFPOSE_IMG_SIZE)
    humans = skeleton_detector.detect(image)
    body_part = humans[0].body_parts[0]
    idx = body_part.part_idx

    txs, tys = draw_tensor_skel(image, humans)

    ####################################################################################################################
    body_estimator = Body(OPENPOSE_MODEL_PATH, device)
    candidate, subset = body_estimator(image)
    pxs, pys = draw_pytorch_skel(image, candidate, subset)

    for tx, ty, px, py in zip(txs, tys, pxs, pys):
        print(tx,px, ty,py)

    cv2.imshow('Temp', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()