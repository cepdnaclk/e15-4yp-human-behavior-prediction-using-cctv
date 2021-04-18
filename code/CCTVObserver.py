import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import datetime
import time
from BehaviorExtraction.BehaviorExtractor import BehaviorExtractor

VIDEO_SRC = 'Demo/CCTV_Low.mp4'

class VideoCapture(QtWidgets.QWidget):
    def __init__(self, filename, parent):
        super(QtWidgets.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(filename)
        self.frameCount = 0
        self.extractor = BehaviorExtractor(1280, 720)
        self.floor = cv2.imread('BehaviorExtraction/assets/Floor_New.png')
        self.floor = cv2.cvtColor(self.floor, cv2.COLOR_BGR2RGB)

        self.video_frame = parent.findChildren(QtWidgets.QLabel,"preview")[0] #QtWidgets.QLabel()

        self.timeLbl = parent.findChildren(QtWidgets.QLabel,"timeLbl")[0]
        self.rateLbl = parent.findChildren(QtWidgets.QLabel, "rateLbl")[0]
        self.fcountLbl = parent.findChildren(QtWidgets.QLabel, "fcountLbl")[0]
        self.pcountLbl = parent.findChildren(QtWidgets.QLabel, "pcountLbl")[0]

        self.recordList = parent.findChildren(QtWidgets.QListWidget, "recordList")[0]

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        tic = time.time()
        frame, pcount, records = self.extractor.detect(frame, self.floor)
        frame = self.get_final_image(frame)
        toc = time.time()
        self.rateLbl.setText(f'Frame Rate     : {round(1 / (toc - tic), 2)}')

        self.frameCount += 1
        self.fcountLbl.setText(f'Frame Count    : {self.frameCount}')
        self.fcountLbl.adjustSize()

        self.pcountLbl.setText(f'People Count   : {pcount}')
        self.pcountLbl.adjustSize()

        for record in records:
            self.recordList.addItem(str(record))
            self.recordList.scrollToBottom()

        date = datetime.datetime.now()
        record = f'{date.strftime("%Y-%m-%d")}  {date.strftime("%H:%M:%S")}'
        self.timeLbl.setText(record)
        self.timeLbl.adjustSize()

        height, width, _ = frame.shape
        img = QtGui.QImage(frame.data, width, height, frame.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

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

        floor_size = self.floor.shape[:2]
        new_floor_size = tuple([int(x * float(size[0]) / floor_size[0]) for x in floor_size])
        new_floor = cv2.resize(self.floor, (new_floor_size[1], new_floor_size[0]))
        return np.concatenate((padded_video, new_floor), axis=1)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()

class CCTVObserver(QMainWindow):
    def __init__(self):
        super(CCTVObserver, self).__init__()
        self.setFixedSize(1600, 850)
        self.setWindowTitle("Person Identifier")

        self.font = QtGui.QFont()
        self.font.setPointSize(15)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setGeometry(0, 0, 1600, 850)
        self.centralwidget.setObjectName("centralwidget")

        self.viewer = QtWidgets.QLabel(self.centralwidget) #ImageViewer(self.centralwidget)
        self.viewer.setObjectName("preview")
        self.viewer.setStyleSheet("background-color: white")
        self.viewer.setGeometry(10,10,1580,590)
        self.viewer.setFixedSize(1580, 590)
        self.initUI()

        self.capture = VideoCapture(VIDEO_SRC, self.centralwidget)


    def initUI(self):
        self.playBtn = QtWidgets.QPushButton(self.centralwidget)
        self.playBtn.setGeometry(10, 610, 141, 51)
        self.playBtn.setText("Play")
        self.playBtn.setFont(self.font)
        self.playBtn.setObjectName("playBtn")
        self.playBtn.clicked.connect(self.startCapture)

        self.stopBtn = QtWidgets.QPushButton(self.centralwidget)
        self.stopBtn.setGeometry(QtCore.QRect(160, 610, 141, 51))
        self.stopBtn.setText("Stop")
        self.stopBtn.setFont(self.font)
        self.stopBtn.setObjectName("stopBtn")

        self.openBtn = QtWidgets.QPushButton(self.centralwidget)
        self.openBtn.setGeometry(QtCore.QRect(310, 610, 141, 51))
        self.openBtn.setText("Open")
        self.openBtn.setFont(self.font)
        self.openBtn.setObjectName("openBtn")

        self.checkPath = QtWidgets.QCheckBox(self.centralwidget)
        self.checkPath.setGeometry(QtCore.QRect(280, 810, 121, 17))
        self.checkPath.setText("Draw Paths")
        self.checkPath.setFont(self.font)
        self.checkPath.setObjectName("checkPath")

        self.checkBoxes = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxes.setGeometry(QtCore.QRect(280, 770, 121, 17))
        self.checkBoxes.setText("Draw Boxes")
        self.checkBoxes.setFont(self.font)
        self.checkBoxes.setObjectName("checkBoxes")

        self.checkSkel = QtWidgets.QCheckBox(self.centralwidget)
        self.checkSkel.setGeometry(QtCore.QRect(280, 730, 151, 17))
        self.checkSkel.setText("Draw Poses")
        self.checkSkel.setFont(self.font)
        self.checkSkel.setObjectName("checkSkel")

        self.rateLbl = QtWidgets.QLabel(self.centralwidget)
        self.rateLbl.setGeometry(QtCore.QRect(10, 730, 241, 16))
        self.rateLbl.setText("Frame Rate     :")
        self.rateLbl.setFont(self.font)
        self.rateLbl.setObjectName("rateLbl")

        self.fcountLbl = QtWidgets.QLabel(self.centralwidget)
        self.fcountLbl.setGeometry(QtCore.QRect(10, 770, 181, 16))
        self.fcountLbl.setText("Frame Count     :")
        self.fcountLbl.setFont(self.font)
        self.fcountLbl.setObjectName("fcountLbl")

        self.pcountLbl = QtWidgets.QLabel(self.centralwidget)
        self.pcountLbl.setGeometry(QtCore.QRect(10, 810, 141, 31))
        self.pcountLbl.setText("People Count   :")
        self.pcountLbl.setFont(self.font)
        self.pcountLbl.setObjectName("pcountLbl")

        self.recordList = QtWidgets.QListWidget(self.centralwidget)
        self.recordList.setGeometry(QtCore.QRect(460, 611, 1131, 231))
        self.recordList.setObjectName("recordList")
        self.recordList.setFont(self.font)

        self.font.setBold(True)
        self.timeLbl = QtWidgets.QLabel(self.centralwidget)
        self.timeLbl.setGeometry(QtCore.QRect(110, 680, 241, 16))
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
    window = CCTVObserver()
    window.show()
    sys.exit(app.exec_())