import glob
import dlib
import numpy as np
import pandas as pd
import math
from PIL import Image, ImageDraw
import face_recognition
import random
import cv2        
from pathlib import Path
import random

dir_list = glob.glob("/Users/chedvi/Desktop/SER517/SER517-Eye-Gaze-Estimation-Models/Eye_Gaze_data")
df = pd.read_csv(dir_list[0]+"log.csv")

file_name = df["count"].tolist()
len(file_name)

leftEyeleft = df["leftEyeleft"].tolist()
leftEyetop = df["leftEyetop"].tolist()
rightEyeright = df["rightEyeright"].tolist()
rightEyebottom = df["rightEyebottom"].tolist()

import glob
import dlib
import numpy as np
import pandas as pd
import math
from PIL import Image, ImageDraw
import face_recognition
import random
import cv2        
from pathlib import Path
import random
from tqdm import tqdm


resolution=64
image_type = "RGB"
basedir = '/Users/chedvi/Desktop/SER517/SER517-Eye-Gaze-Estimation-Models/Eye_Gaze_data'


left_eye = []
right_eye = []
gaze_point = []
left_eye_right_top = []
left_eye_left_bottom = []
right_eye_right_top = []
right_eye_left_bottom = []
euler = []
face_grid = []
left_eye_grid = []
right_eye_grid = []
facepos = []

dir_name = basedir + target
df = pd.read_csv(dir_name+"log.csv")
file_name = df["count"].tolist()
im = Image.open(dir_name+"lefteye/"+str(file_name[0]).zfill(5)+".jpg").convert(image_type)
gazeX = df["gazeX"].tolist()
gazeY = df["gazeY"].tolist()
eulerX = df["eulerX"].tolist()
eulerY = df["eulerY"].tolist()
eulerZ = df["eulerZ"].tolist()
faceX = df["faceX"].tolist()
faceY = df["faceY"].tolist()
leftEyeleft = df["leftEyeleft"].tolist()
leftEyetop = df["leftEyetop"].tolist()
leftEyeright = df["leftEyeright"].tolist()
leftEyebottom = df["leftEyebottom"].tolist()
rightEyeleft = df["rightEyeleft"].tolist()
rightEyetop = df["rightEyetop"].tolist()
rightEyeright = df["rightEyeright"].tolist()
rightEyebottom = df["rightEyebottom"].tolist()
