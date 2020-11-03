import cv2
import time

def detectPose():

    protoFile = "../Assets/Models/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "../Assets/Models/pose_iter_160000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    print("[INFO]: Using GPU device")
    # additional variables
    threshold = 0.1
    inWidth = 368
    inHeight = 368
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],[11, 12], [12, 13]]

    print("[INFO]: Start Video Capture")
    video = cv2.VideoCapture('../Assets/Videos/common2.mp4')

    while True:
        t = time.time()
        check, frame = video.read()
        if check:
            frame = cv2.resize(frame, (0, 0), cv2.INTER_AREA, 0.5, 0.5)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()

            H = output.shape[2]
            W = output.shape[3]

            # Empty list to store the detected keypoints
            points = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > threshold:
                    #cv2.circle(frameCopy, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                    cv2.circle(frame, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            fps = round(1 / (time.time() - t), 2)
            cv2.putText(frame, f'FPS:{fps}', (6, 18), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow('Output-Skeleton', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

detectPose()