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



for i in tqdm(range(len(file_name))):
    left_eye_image = np.asarray(Image.open(dir_name+"lefteye/"+str(file_name[i]).zfill(5)+".jpg").convert(image_type).resize((resolution,resolution)))/255
    right_eye_image = np.asarray(Image.open(dir_name+"righteye/"+str(file_name[i]).zfill(5)+".jpg").convert(image_type).resize((resolution,resolution)))/255
    left_eye.append(left_eye_image)
    right_eye.append(right_eye_image)
    facegrid = np.genfromtxt (dir_name+"facegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    face_grid.append(facegrid)
    lefteyegrid = np.genfromtxt (dir_name+"lefteyegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    left_eye_grid.append(lefteyegrid)
    righteyegrid = np.genfromtxt (dir_name+"righteyegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    right_eye_grid.append(righteyegrid)

    gaze_point.append([float(gazeX[i]),float(gazeY[i])])
    euler.append([float(eulerX[i]), float(eulerY[i]), float(eulerZ[i])])
    facepos.append([float(faceX[i]), float(faceY[i])])
    left_eye_right_top.append([float(leftEyeright[i]), float(leftEyetop[i])])
    left_eye_left_bottom.append([float(leftEyeleft[i]), float(leftEyebottom[i])])
    right_eye_right_top.append([float(rightEyeright[i]), float(rightEyetop[i])])
    right_eye_left_bottom.append([float(rightEyeleft[i]), float(rightEyebottom[i])])
        
left_eye = np.asarray(left_eye)
right_eye = np.asarray(right_eye)
gaze_point = np.asarray(gaze_point)
face_grid = np.asarray(face_grid)
left_eye_grid = np.asarray(left_eye_grid)
right_eye_grid = np.asarray(right_eye_grid)
euler = np.asarray(euler)
facepos = np.asarray(facepos)
left_eye_right_top = np.asarray(left_eye_right_top)
left_eye_left_bottom = np.asarray(left_eye_left_bottom)
right_eye_right_top = np.asarray(right_eye_right_top)
right_eye_left_bottom = np.asarray(right_eye_left_bottom)


save_dir="/Gazel_prep_data/"+image_type+"Data/"+target
Path(save_dir).mkdir(parents=True, exist_ok=True)
             
#save to File
np.save(save_dir+"gaze_point.npy",gaze_point)
np.save(save_dir+"left_eye.npy",left_eye)
np.save(save_dir+"right_eye.npy",right_eye)
np.save(save_dir+"face_grid.npy",face_grid)
np.save(save_dir+"left_eye_grid.npy",left_eye_grid)
np.save(save_dir+"right_eye_grid.npy",right_eye_grid)
np.save(save_dir+"euler.npy",euler)
np.save(save_dir+"facepos.npy",facepos)
np.save(save_dir+"left_eye_right_top.npy",left_eye_right_top)
np.save(save_dir+"left_eye_left_bottom.npy",left_eye_left_bottom)
np.save(save_dir+"right_eye_right_top.npy",right_eye_right_top)
np.save(save_dir+"right_eye_left_bottom.npy",right_eye_left_bottom)