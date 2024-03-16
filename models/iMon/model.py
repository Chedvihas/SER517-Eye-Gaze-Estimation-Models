import tensorflow as tf
if (tf.version.VERSION == '2.3.0'):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, SeparableConv2D, DepthwiseConv2D, Input, GlobalMaxPooling2D, Activation, Conv2D, Conv3D, Reshape, AveragePooling3D, AveragePooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D, LSTM, Embedding, Dense, Dropout, Flatten, BatchNormalization, add, UpSampling2D, Conv2DTranspose
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from losses import euclidean_loss, heatmap_loss
import config


def conv_bn(input_x, conv):
    x = conv(input_x)
    x = Activation('relu')(x)
    bn_name = 'single_v10_' + input_x.name + '_' + conv.name + '_bn'
    bn_name = bn_name.replace(':', '_')
    x = BatchNormalization(name=bn_name)(x)
    return x


def relu_bn(input_x):
    x = Activation('relu')(input_x)
    bn_name = 'single_v10_' + input_x.name + '_bn'
    bn_name = bn_name.replace(':', '_')
    x = BatchNormalization(name=bn_name)(x)
    return x


policy = mixed_precision.Policy('mixed_float16')
if (tf.version.VERSION == '2.3.0'):
    mixed_precision.set_policy(policy)
else:
    mixed_precision.set_global_policy(policy)


grid_dense1 = Dense(256, activation="relu", name='grid_dense1')
grid_dense2 = Dense(128, activation="relu", name='grid_dense2')
grid_dense3 = Dense(128, activation="relu", name='grid_dense3')
heatmap_conv1 = Conv2D(1, (7, 7), padding='same', name='heatmap_conv1')
heatmap_conv2 = Conv2D(1, (7, 7), padding='same', name='heatmap_conv2')
heatmap_conv3 = Conv2D(1, (3, 3), padding='same', name='heatmap_conv3')

