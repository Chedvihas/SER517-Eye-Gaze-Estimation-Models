import numpy as np
target="//PreProcessedData/Approach1-Data/Output/train/images/"

left_eye_right_top = np.load(target+"left_eye_right_top.npy")
left_eye_left_bottom = np.load(target+"left_eye_left_bottom.npy")
right_eye_right_top = np.load(target+"right_eye_right_top.npy")
right_eye_left_bottom = np.load(target+"right_eye_left_bottom.npy")


left_eye_right_top[:,1] = left_eye_right_top[:,1] - left_eye_left_bottom[:,1]
left_eye_right_top[:,0] = left_eye_left_bottom[:,0] - left_eye_right_top[:,0]

right_eye_right_top[:,1] = right_eye_right_top[:,1] - right_eye_left_bottom[:,1]
right_eye_right_top[:,0] = right_eye_left_bottom[:,0] - right_eye_right_top[:,0]

left_eye_size = left_eye_right_top
right_eye_size = right_eye_right_top

import json
import glob
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import concatenate, ZeroPadding2D, Add, add, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model
import gc
from pathlib import Path

dropout_rate=0
resolution = 64
channels = 1
target="/special/jbpark/TabS6LData/Joonbeom/train_dataset/"
model_dir = "/special/jbpark/EvalModel"

def custom_loss(y_true, y_pred): 
    #euclidean loss
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) # /205.7
    #squared loss
    #return K.sum(K.square(y_pred - y_true), axis=-1) 
    
SConv1 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')
SConv2 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
SConv3 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')

# Left Eye
input1 = Input(shape=(resolution, resolution,channels), name='left_eye')
x = SConv1(input1)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv2(x)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv3(x)
left_eye = MaxPooling2D(pool_size = (2, 2))(x)
left_eye = Flatten()(left_eye)

# Right Eye
input2 = Input(shape=(resolution, resolution,channels), name='right_eye')
x = SConv1(input2)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv2(x)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

x = SConv3(x)
right_eye = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
right_eye = Flatten()(right_eye)

