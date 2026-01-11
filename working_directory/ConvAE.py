from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPool2D
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf


def CAE(input_shape=(224, 224, 1), classes=2, num_feature=20, transfer_learning=False, tlmodel_path=''):
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

    if transfer_learning:
        pretrained_model = tf.keras.models.load_model(tlmodel_path)
        
        weights = pretrained_model.get_layer('conv2d').get_weights()
        model.get_layer('conv2d').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_1').get_weights()
        model.get_layer('conv2d_1').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_2').get_weights()
        model.get_layer('conv2d_2').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_3').get_weights()
        model.get_layer('conv2d_3').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_4').get_weights()
        model.get_layer('conv2d_4').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_5').get_weights()
        model.get_layer('conv2d_5').set_weights(weights)
        weights = pretrained_model.get_layer('embedding').get_weights()
        model.get_layer('embedding').set_weights(weights)
        weights = pretrained_model.get_layer('dense').get_weights()
        model.get_layer('dense').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose').get_weights()
        model.get_layer('conv2d_transpose').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_1').get_weights()
        model.get_layer('conv2d_transpose_1').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_2').get_weights()
        model.get_layer('conv2d_transpose_2').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_3').get_weights()
        model.get_layer('conv2d_transpose_3').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_4').get_weights()
        model.get_layer('conv2d_transpose_4').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_5').get_weights()
        model.get_layer('conv2d_transpose_5').set_weights(weights)
        weights = pretrained_model.get_layer('conv2d_transpose_6').get_weights()
        model.get_layer('conv2d_transpose_6').set_weights(weights)
        
        for layer in model.layers:
            layer.trainable = False
        model.get_layer('embedding').trainable = True
        model.get_layer('dense').trainable = True
        
        print("transfer learning complete=============================================")

    model.summary()
    return model