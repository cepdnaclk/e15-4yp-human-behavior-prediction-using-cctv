import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import datetime
import time
from PersonIdentification.PersonIdentifier import PersonIdentifier

VIDEO_SRC = 'Demo/Entrance_2.mp4'

class VideoCapture(QtWidgets.QWidget):
    def __init__(self, filename, parent):
        super(QtWidgets.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(filename)
        self.frameCount = 0
        self.identifier = PersonIdentifier(1280, 720)

        self.mainView = parent.findChildren(QtWidgets.QLabel,"mainView")[0] #QtWidgets.QLabel()
        self.humanView = parent.findChildren(QtWidgets.QLabel, "humanView")[0]  # QtWidgets.QLabel()

        self.timeLbl = parent.findChildren(QtWidgets.QLabel,"timeLbl")[0]
        self.rateLbl = parent.findChildren(QtWidgets.QLabel, "rateLbl")[0]
        self.fcountLbl = parent.findChildren(QtWidgets.QLabel, "fcountLbl")[0]
        self.pcountLbl = parent.findChildren(QtWidgets.QLabel, "pcountLbl")[0]


    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        tic = time.time()
        frame, humanFrame = self.identifier.detect(frame)
        frame = self.get_final_image(frame)
        toc = time.time()
        self.rateLbl.setText(f'Frame Rate     : {round(1 / (toc - tic), 2)}')

        self.frameCount += 1
        self.fcountLbl.setText(f'Frame Count    : {self.frameCount}')
        self.fcountLbl.adjustSize()

        #self.pcountLbl.setText(f'People Count   : {pcount}')
        #self.pcountLbl.adjustSize()

        date = datetime.datetime.now()
        record = f'{date.strftime("%Y-%m-%d")}  {date.strftime("%H:%M:%S")}'
        self.timeLbl.setText(record)
        self.timeLbl.adjustSize()

        height, width, _ = frame.shape
        img = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.mainView.setPixmap(pix)


        h, w, _ = humanFrame.shape
        img2 = QtGui.QImage(humanFrame.data, w, h, humanFrame.strides[0], QtGui.QImage.Format_RGB888)
        pix2 = QtGui.QPixmap.fromImage(img2)
        self.humanView.setPixmap(pix2)

    def get_final_image(self, video, size=[590, 1048]):  # (height, width) format
        video_size = video.shape[:2]
        ratio_h = size[0] / video_size[0]
        ratio_w = size[1] / video_size[1]

        new_video_size = tuple([int(i * ratio_h) for i in video_size])

        resized_video = cv2.resize(video, (new_video_size[1], new_video_size[0]))

        delta_w = size[1] - new_video_size[1]
        delta_h = size[0] - new_video_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded_video = cv2.copyMakeBorder(resized_video, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_video

    def start(self):
        self.timer = QtCore.QTimer()
        #self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()

class FaceObserver(QMainWindow):
    def __init__(self):
        super(FaceObserver, self).__init__()
        self.setFixedSize(1600, 850)
        self.setWindowTitle("Person Identifier")

        self.font = QtGui.QFont()
        self.font.setPointSize(15)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setGeometry(0, 0, 1600, 850)
        self.centralwidget.setObjectName("centralwidget")

        self.mainView = QtWidgets.QLabel(self.centralwidget) #ImageViewer(self.centralwidget)
        self.mainView.setObjectName("mainView")
        self.mainView.setStyleSheet("background-color: black")
        self.mainView.setGeometry(QtCore.QRect(10, 10, 1131, 581))
        self.mainView.setFixedSize(1131, 581)

        self.humanView = QtWidgets.QLabel(self.centralwidget)  # ImageViewer(self.centralwidget)
        self.humanView.setObjectName("humanView")
        self.humanView.setStyleSheet("background-color: black")
        self.humanView.setGeometry(QtCore.QRect(10, 600, 1581, 241))
        self.humanView.setFixedSize(1581, 241)

        self.initUI()
        self.capture = VideoCapture(VIDEO_SRC, self.centralwidget)

    def initUI(self):
        self.playBtn = QtWidgets.QPushButton(self.centralwidget)
        self.playBtn.setGeometry(QtCore.QRect(1150, 10, 221, 51))
        self.playBtn.setText("Play")
        self.playBtn.setFont(self.font)
        self.playBtn.setObjectName("playBtn")
        self.playBtn.clicked.connect(self.startCapture)

        self.stopBtn = QtWidgets.QPushButton(self.centralwidget)
        self.stopBtn.setGeometry(QtCore.QRect(1380, 10, 211, 51))
        self.stopBtn.setText("Stop")
        self.stopBtn.setFont(self.font)
        self.stopBtn.setObjectName("stopBtn")

        self.openBtn = QtWidgets.QPushButton(self.centralwidget)
        self.openBtn.setGeometry(QtCore.QRect(1150, 70, 441, 51))
        self.openBtn.setText("Open")
        self.openBtn.setFont(self.font)
        self.openBtn.setObjectName("openBtn")

        self.rateLbl = QtWidgets.QLabel(self.centralwidget)
        self.rateLbl.setGeometry(QtCore.QRect(1160, 230, 241, 16))
        self.rateLbl.setText("Frame Rate     :")
        self.rateLbl.setFont(self.font)
        self.rateLbl.setObjectName("rateLbl")

        self.fcountLbl = QtWidgets.QLabel(self.centralwidget)
        self.fcountLbl.setGeometry(QtCore.QRect(1160, 270, 181, 16))
        self.fcountLbl.setText("Frame Count     :")
        self.fcountLbl.setFont(self.font)
        self.fcountLbl.setObjectName("fcountLbl")

        self.pcountLbl = QtWidgets.QLabel(self.centralwidget)
        self.pcountLbl.setGeometry(QtCore.QRect(1160, 300, 141, 31))
        self.pcountLbl.setText("People Count   :")
        self.pcountLbl.setFont(self.font)
        self.pcountLbl.setObjectName("pcountLbl")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1160, 440, 181, 16))
        self.label_5.setText("Distance Threshold :")
        self.label_5.setFont(self.font)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1160, 390, 181, 16))
        self.label_6.setText("Face Threshold :")
        self.label_6.setFont(self.font)
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1160, 480, 171, 31))
        self.label_7.setText("Save Time Limit :")
        self.label_7.setFont(self.font)
        self.label_7.setObjectName("label_7")

        self.fthresEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.fthresEdit.setGeometry(QtCore.QRect(1360, 380, 111, 31))
        self.fthresEdit.setFont(self.font)
        self.fthresEdit.setObjectName("fthresEdit")

        self.distEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.distEdit.setGeometry(QtCore.QRect(1360, 430, 111, 31))
        self.distEdit.setFont(self.font)
        self.distEdit.setObjectName("distEdit")

        self.stimeEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.stimeEdit.setGeometry(QtCore.QRect(1360, 480, 111, 31))
        self.stimeEdit.setFont(self.font)
        self.stimeEdit.setObjectName("stimeEdit")

        self.font.setBold(True)
        self.timeLbl = QtWidgets.QLabel(self.centralwidget)
        self.timeLbl.setGeometry(QtCore.QRect(1260, 160, 241, 16))
        self.timeLbl.setFont(self.font)
        self.timeLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.timeLbl.setObjectName("timeLbl")
        date = datetime.datetime.now()
        record = f'{date.strftime("%Y-%m-%d")}  {date.strftime("%H:%M:%S")}'
        self.timeLbl.setText(record)

    def startCapture(self):
        if self.capture is not None:
            self.capture.start()
            self.stopBtn.clicked.connect(self.capture.stop)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceObserver()
    window.show()
    sys.exit(app.exec_())