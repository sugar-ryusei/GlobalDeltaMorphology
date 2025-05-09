from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPool2D
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf


def CAE(input_shape=(224, 224, 1), classes=2, num_feature=20, tlmodel_path=''):
    filters=[32, 64, 128, 256, 512, 1024, num_feature]
    model = Sequential()
    deep = 2**(len(filters)-1)
    if input_shape[0] % deep == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu'))
    
    model.add(Conv2D(filters[2], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2D(filters[3], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2D(filters[4], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2D(filters[5], 3, strides=2, padding=pad3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(units=filters[6], name='embedding'))
    model.add(Dense(units=filters[5]*int(input_shape[0]/deep)*int(input_shape[0]/deep), activation='relu'))

    model.add(Reshape((int(input_shape[0]/deep), int(input_shape[0]/deep), filters[5])))
    model.add(Conv2DTranspose(filters[4], 3, strides=2, padding=pad3, activation='relu'))

    model.add(Conv2DTranspose(filters[3], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2DTranspose(filters[2], 5, strides=2, padding='same', activation='relu'))
    
    model.add(Conv2DTranspose(filters[1], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu'))

    model.add(Conv2DTranspose(classes, 1, strides=1, padding='same', activation='softmax'))

    model.summary()
    return model