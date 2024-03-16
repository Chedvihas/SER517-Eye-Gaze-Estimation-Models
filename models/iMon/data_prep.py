import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate, SeparableConv2D, DepthwiseConv2D, Input, GlobalMaxPooling2D, Activation, Conv2D, Conv3D, Reshape, AveragePooling3D, AveragePooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D, LSTM, Embedding, Dense, Dropout, Flatten, BatchNormalization, add, UpSampling2D, Conv2DTranspose
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import generator
from sklearn.utils import shuffle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def get_gram_matrix_loss(pred, label):
    pred_gram = gram_matrix(pred)
    label_gram = gram_matrix(label)
    return K.mean(K.abs(pred_gram - label_gram))


def get_resnet_model():
    model = ResNet50V2(weights='imagenet', include_top=False)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    layer_indices = [9, 36, 82, 150, 185]  # activation indices ~ scale 0.015
    layer_indices = [37, 83, 151]  # max_pool indices ~ scale 0.018
    outputs = None
    for idx in layer_indices:
        features = model.layers[idx].output
        features = Flatten()(features)
        if (idx == layer_indices[0]):
            outputs = features
        else:
            outputs = concatenate([outputs, features])
    resnet_model = Model(model.inputs, outputs)
    return resnet_model


