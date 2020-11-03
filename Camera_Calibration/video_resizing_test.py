import cv2

cap= cv2.VideoCapture('rtsp://admin:abcd%401234@192.168.1.104/Streaming/Channels/101')


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)
    
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_480p()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while(True):
    ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    frame75 = rescale_frame(frame, percent=35)
    cv2.imshow('frame75', frame75)
    # frame150 = rescale_frame(frame, percent=150)
    # cv2.imshow('frame150', frame150)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()