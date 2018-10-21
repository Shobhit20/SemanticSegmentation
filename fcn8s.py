# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Activation
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def skip_connections(skip_conn1, skip_conn2, skip_conn3, nClasses):
    n = 4096
    IMAGE_ORDERING = "channels_last"
    decoder = Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6")(skip_conn1)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(decoder)

    # Deconvolution Layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                              data_format=IMAGE_ORDERING)(conv7)

    pool411 = Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11",
                     data_format=IMAGE_ORDERING)(skip_conn2)
    pool411_2 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
                                data_format=IMAGE_ORDERING)(pool411)

    pool311 = Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11",
                     data_format=IMAGE_ORDERING)(skip_conn3)

    decoder = Add(name="add")([pool411_2, pool311, conv7_4])
    decoder = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(decoder)
    decoder = (Activation('softmax'))(decoder)
    return decoder


def VGG16_backbone(nClasses=21):

    img_input = Input(shape=( 224, 224, 3))


    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    skip_conn_pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    skip_conn_pool2 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    skip_conn_pool1 = x

    model3 = Model(img_input, x, name='vgg16')

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')

    model3.load_weights(weights_path)

    decoder_architecture = skip_connections(skip_conn_pool1, skip_conn_pool2, skip_conn_pool3, nClasses)

    model = Model(img_input, decoder_architecture)

    return model


