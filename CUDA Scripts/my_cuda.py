import time
import cv2 as cv
import numpy as np
from numba import cuda


def gpuDeviceInfo(device):
    cv.cuda.printShortCudaDeviceInfo(device)


gpuDeviceInfo(0)