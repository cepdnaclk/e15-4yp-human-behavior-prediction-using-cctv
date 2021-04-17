import numpy as np
import cv2

joints = [[1, 2], [1, 5], [2, 3],  [3, 4],  [5, 6],  [6, 7],
          [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],[12, 13],
          [0, 1], [0, 14],[14, 16],[0, 15], [15, 17]]

jNew = [[1, 2], [1, 5], [2, 3],  [3, 4],  [5, 6],  [6, 7]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

'''
def draw_skeleton(canvas, skeletons, z1, z2):
    for skeleton in skeletons:
        i = 0
        for x, y in skeleton.values():
            cv2.putText(canvas, str(i), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, [0, 255, 0], 1,lineType=cv2.LINE_AA)
            cv2.circle(canvas, (z1+int(x)*2, z2+int(y)*2), 4,  colors[i], thickness=-1)
            i += 1

        print(skeleton.keys())
        i = 0
        for idx1, idx2 in joints:
            if idx1 in skeleton.keys() and idx2 in skeleton.keys():
                x1, y1 = skeleton[idx1]
                x2, y2 = skeleton[idx2]
                #print(idx1, idx2, int(x1), int(y1), int(x2), int(y2))
                cv2.line(canvas, (z1+int(x1)*2, z2+int(y1)*2), (z1+int(x2)*2 , z2+int(y2)*2), colors[i], 2)
                i += 1
'''

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset, z1, z2):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (z1+int(x)*2, z2+int(y)*2), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            cv2.line(canvas, (z1+int(Y[0]*2), z2+int(X[0]*2)), (z1+int(Y[1]*2) , z2+int(X[1]*2)), colors[i], 2)

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j