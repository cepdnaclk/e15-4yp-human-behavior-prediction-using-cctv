from PIL import Image, ImageTk
import tkinter as tk
import argparse
import cv2
import time
import numpy as np



class Application:
    def __init__(self, output_path = "./"):
        
        self.CHECKERBOARD = (6,9)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = [] 


        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.prev_img_shape = None
        
        
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        #initializing the timer value
        self.counter = 0        
       
        self.no_snapshots = 0
        self.initial_time = 0
        
        self.snapshot_int_count = 0
        self.time_int_val = 0
        
        self.is_calibrate = False
        self.is_calibrate_done = False
        
        
        # self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.vs = cv2.VideoCapture('rtsp://admin:abcd%401234@192.168.1.107/Streaming/Channels/101')
        
        #self.vs = cv2.VideoCapture('path 4 - full human - double - square.mp4') # capture video frames, 0 is your default video camera
       
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window           
        
        self.snapshot_count = tk.StringVar()
        self.time_val = tk.StringVar()
        self.snapshot_count.set("20")
        self.time_val.set("5")
        
        
        self.root.title("PyImageSearch PhotoBooth")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel1 = tk.Label(self.root)  # initialize image panel
        #self.panel1.pack(padx=10, pady=10)
        self.panel1.grid(row = 1, column = 1)
        
        self.panel2 = tk.Label(self.root )  # initialize image panel
        #self.panel2.pack(padx=10, pady=10)
        self.panel2.grid(row = 1, column = 2)        
        
        
        #create a Text label for two panels
        self.text1 = tk.Text(self.root, height=1, width=15)
        self.text1.grid(row = 2 , column = 1)
        self.text1.insert(tk.END, "Original Feed")
        
        self.text1 = tk.Text(self.root, height=1, width=15)
        self.text1.grid(row = 2 , column = 2)
        self.text1.insert(tk.END, "Calibrated Feed")
        
        #create an entry field for snapshots
        self.entry_label_snapshots = tk.Label(self.root, text = "Enter the number of snapshots requird : ")
        self.entry_label_snapshots.grid(row = 3 , column = 1 , columnspan = 2, sticky= "W" ,  padx = 525 )
        
        self.entry_field_snapshots = tk.Entry(self.root , textvariable = self.snapshot_count, width = 4)
        self.entry_field_snapshots.grid(row = 3, column = 1 , columnspan = 2, sticky = "E" , padx = 520 )
   
        
        #create an entry field for time interval
        self.entry_label_time = tk.Label(self.root, text = "Enter the time interval : ")
        self.entry_label_time.grid(row = 4 , column = 1 , columnspan = 2,  sticky= "W" ,  padx = 525 )
        
        self.entry_field_time = tk.Entry(self.root, textvariable = self.time_val , width = 4 )
        self.entry_field_time.grid(row = 4, column = 1 , columnspan = 2, sticky = "E", padx = 520)
        
        
        #create a label  to show the number of  valid snapshots + time remaining for the next shot        
        self.snapshot_label1 = tk.Label(self.root, text="No snapshots yet")
        self.snapshot_label1.grid(row = 6 , column = 1 , columnspan = 2)
        
        self.snapshot_label2 = tk.Label(self.root, text="No. of seconds remaining to the next shot : 0")        
        self.snapshot_label2.grid(row = 7 ,column = 1 , columnspan = 2)
        
     
        # create a button, that when pressed, will take the current frame and save it to file
        btn = tk.Button(self.root, text="Calibrate!", command=self.cam_calibrate)
        #btn.pack(side = 'left',fill="both", expand=True, padx=10, pady=10)
        btn.grid(row = 5 , column = 1 , columnspan = 2)  
        
        
              
        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        current_time = time.time()
        elapsed_time = (int)(current_time - self.initial_time)
        
        ok, frame = self.vs.read()  # read frame from video stream        
        if ok:  # frame captured without any errors
            width = int(frame.shape[1] * 55 / 100)
            height = int(frame.shape[0] *55 /100)
            
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            
            
            self.panel1.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel1.config(image=imgtk)  # show the image
            
            if self.is_calibrate_done:
                print(self.mtx)
                print("`````````````````````````````````````````````````")
                print(self.dist)
                print("`````````````````````````````````````````````````")
                print(self.newcameramtx)
                undst = cv2.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
                newframe = cv2.resize(undst, dim, interpolation =cv2.INTER_AREA)
                cv2image = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
                current_modify_image = Image.fromarray(cv2image)  # convert image for PIL
                imgtk2 = ImageTk.PhotoImage(image=current_modify_image)  # convert image for tkinter
                self.panel2.imgtk = imgtk2  # anchor imgtk so it does not be deleted by garbage-collector
                self.panel2.config(image=imgtk2)  # show the image
                
            else:
                blank_image = np.ones((height,width,3),np.uint8)
                blank_image[:] = (0, 0, 255)
                blank_image = Image.fromarray(blank_image)
                imgtk2 = ImageTk.PhotoImage(image=blank_image)
                self.panel2.imgtk = imgtk2  # anchor imgtk so it does not be deleted by garbage-collector
                self.panel2.config(image=imgtk2)  # show the image
                
            
        if self.is_calibrate:
            if elapsed_time >= 1:
                self.initial_time = current_time         
                self.snapshot_label2['text'] = "No. of seconds remaining to the next shot : " + str(self.counter)                
                self.snapshot_label2.update()
        
                if (self.counter == 0) and (self.no_snapshots < self.snapshot_int_count ):
                    time.sleep(1)
                    self.initial_time = time.time()                
                                        
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                   
                    # Find the chess board corners
                    # If desired number of corners are found in the image then ret = true
                    ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                    
                    if ret == True:
                        self.objpoints.append(self.objp)
                        # refining pixel coordinates for given 2d points.
                        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), self.criteria)        
                        self.imgpoints.append(corners2)
                        
                        self.no_snapshots += 1
                        self.snapshot_label1['text'] = "No. of snapshots : " + str(self.no_snapshots)
                        self.snapshot_label1.update()            
                        
                        
                    self.counter = self.time_int_val + 1
                    
                self.counter -= 1
                
            if self.no_snapshots == self.snapshot_int_count:
                self.no_snapshots = 0
                self.is_calibrate = False
                self.counter = 0;
                self.snapshot_label2['text'] = "No. of seconds remaining to the next shot : " + str(self.counter)
                self.snapshot_label2.update()
                self.snapshot_label1['text'] = " Calibration Completed " 
                self.snapshot_label1.update()
                
                h,w = frame.shape[:2]

                """
                Performing camera calibration by 
                passing the value of known 3D points (objpoints)
                and corresponding pixel coordinates of the 
                detected corners (imgpoints)
                """
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
                
                mean_error = 0
                for i in range(len(self.objpoints)):
                    imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    mean_error += error
                print( "total error: {}".format(mean_error/len(self.objpoints)) )
                
                print("Camera matrix : \n")
                print(mtx)
                print("dist : \n")
                print(dist)
                print("rvecs : \n")
                print(rvecs)
                print("tvecs : \n")
                print(tvecs)
                
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
                
                cv_file = cv2.FileStorage("camera_calibration.xml", cv2.FILE_STORAGE_WRITE)
                cv_file.write("camera_matrix", mtx)
                cv_file.write("distortion_matrix", dist)
                cv_file.write("new_camera_matrix", newcameramtx)
                cv_file.write("roi", roi)
                # note you *release* you don't close() a FileStorage object
                cv_file.release()  
                
                self.is_calibrate_done = True
                self.load_values()
                
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds
    
    def load_values(self):
        cv_file = cv2.FileStorage("camera_calibration.xml", cv2.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode("camera_matrix").mat()
        self.dist = cv_file.getNode("distortion_matrix").mat()
        self.newcameramtx = cv_file.getNode("new_camera_matrix").mat() 
        print("[INFO]  All files are now loaded!")
        cv_file.release()       
    
   

    def cam_calibrate(self):
        """ Start the calibration procedure """        
        temp_string = self.entry_field_snapshots.get()
        self.snapshot_int_count = int(temp_string.strip())
        
        temp_string = self.entry_field_time.get()
        self.time_int_val = int(temp_string.strip())
        
        self.no_snapshots = 0
        self.initial_time = time.time()
        self.is_calibrate = True
        self.is_calibrate_done = False
        self.counter = self.time_int_val
        
      
        
        
    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()
