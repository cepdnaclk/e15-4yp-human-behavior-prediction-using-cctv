import cv2

cv_file = cv2.FileStorage("camera_calibration.xml", cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("distortion_matrix").mat()
newcameramtx = cv_file.getNode("new_camera_matrix").mat() 
print("[INFO]  All files are now loaded!")
cv_file.release()  
     
cap = cv2.VideoCapture('rtsp://admin:abcd%401234@192.168.1.107/Streaming/Channels/101')
#cap = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1280,720))
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    out.write(dst)
    
    #dst = dst[y:y+h, x:x+w]
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()