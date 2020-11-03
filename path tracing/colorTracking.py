import cv2
import numpy as np



def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        #hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            print(area)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)           
            
            if objCor > 4  and objCor < 12: 
                objectType = "circle"
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
                
            
                # provide a point you wish to map from image 1 to image 2
                a = np.array([[(x + w/2), (y+ h/2)]], dtype='float32')
                a = np.array([a])
            
                # finally, get the mapping
                pointsOut = cv2.perspectiveTransform(a, h_mat)
                print(x + w/2)
                print(y + h/2)            
                print(pointsOut[0][0])
            
           
                pts = np.array([[140, 678], [336,259], [385,147],[736,80], [916 , 35] , [1113 , 571]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(imgContour,[pts],True,(0,0,255))
                return pointsOut
            
  
#cap = cv2.VideoCapture('rtsp://admin:abcd%401234@192.168.1.103/Streaming/Channels/101')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('path 7 - object - custom path .mp4') # capture video frames, 0 is your default video camera


# provide points from image 1
pts_src = np.array([[140, 678], [336,259], [385,147],[736,80], [916 , 35] , [1113 , 571]])

# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
pts_dst = np.array([[0,335],[0,104],[0,0],[289,0], [406,0] , [335,406]])
#pts_dst = np.array([[335 ,0],[104, 0],[0,0],[0,289], [0,406] , [406,335]])

# calculate matrix H
h_mat, status = cv2.findHomography(pts_src, pts_dst)

# imgpath = np.zeros((406,335,3) , np.uint8)
# imgpath[:] = (29, 101,181)

imgpath = cv2.imread('Resources/pathblank.png',3)

Warning
cur_point = [0, 0]
pre_point = [0, 0]
got_point = False

lower = np.array([0, 82 , 109])
upper =np.array([179, 255, 255]) 

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    h,  w = frame.shape[:2]

    newpath = imgpath.copy()
    img = frame
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)         
    mask  = cv2.inRange(imgHSV , lower , upper)
    imgResult = cv2.bitwise_and(img, img , mask = mask)
    
    imgGray = cv2.cvtColor(imgResult , cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray , (7,7) , 1)
    imgCanny = cv2.Canny(imgBlur , 50,50)
    # cv2.imshow("original" , frame)   
    # cv2.imshow('Result', imgBlur)
    
    imgBlank2 = np.zeros_like(img)
    
    imgContour = img.copy()
    temp = getContours(imgCanny)
    if temp is not None:
        cur_point = temp[0][0]
        if got_point:
            cv2.line(newpath , (pre_point[0] , pre_point[1]) , (cur_point[0] , cur_point[1]) , (255, 255, 0) , 5)
            print("drawq")
            print("--------------------------------------------------------------")
        else:
            got_point = True
            
        pre_point = cur_point
    
    imgpath = newpath
    frame = cv2.resize(frame , (640 , 480))
    #imgStack = stackImages(0.6 , ([frame, imgContour , imgBlur] , [ imgCanny  , imgpath,  imgBlank2 ]))
    cv2.imshow('Original', imgContour)
    cv2.imshow('Result', imgpath)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()