import os
import numpy as np
import scipy.io as sio

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from imutils import face_utils
import imutils
import dlib
import cv2

from ITrackerModel import ITrackerModel

CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename, map_location=torch.device('cpu'))
    return state

def getFaceGrid(frameW, frameH, faceWidth, faceLeft, faceTop, faceHeight):
    gridW = 25
    gridH = 25
    scaleX = gridW / frameW
    scaleY = gridH / frameH

    grid = np.zeros((gridH, gridW))

    xLo = round(faceLeft * scaleX) + 1
    yLo = round(faceTop * scaleY) + 1
    w = round(faceWidth * scaleX)
    h = round(faceHeight * scaleY)
    xHi = xLo + w - 1
    yHi = yLo + h - 1

    xLo = min(gridW, max(1, xLo))
    xHi = min(gridW, max(1, xHi))
    yLo = min(gridH, max(1, yLo))
    yHi = min(gridH, max(1, yHi))

    grid[yLo: yHi, xLo: xHi] = np.ones((yHi - yLo, xHi - xLo))

    grid = np.asmatrix(grid)
    grid = grid.getH()
    grid = grid[:].getH()
    labelFaceGrid = grid
    return labelFaceGrid

