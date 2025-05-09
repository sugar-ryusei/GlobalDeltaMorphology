import numpy as np
import os
import cv2
import glob
import tifffile

#import time
import math
import tensorflow as tf
import random

class DatasetTF(object):

    def __init__(self,
                 batch_size=16,
                 augmentation=False,
                 data_path='delta_images/*.tiff'
                 ):
    
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.data_paths = data_path
        self.init_len_data = len(self.data_paths)
        self.data_length = len(self.data_paths)
        

    def load_label(self, image_file):
        image = image_file
        image = tf.cast(image, tf.int64)
        image = self.normalize(image)
        image = self.to_one_hot(image)
        return image

    def load_data_label(self, im_path, la_path):
        data = im_path
        label = self.load_label(la_path)
        return data, label

    def normalize(self, image):
        return image
    
    def im2vec(self, filename):
        im = tifffile.imread(filename).astype(np.float32).reshape([224,224,1])
        im = im[:,:,0]
        im = im.ravel(order='F')
        return im

    def set_classes(self):
        tmp = [self.im2vec(i) for i in self.data_paths]
        im = np.concatenate(tmp)
        
        im_u, con = np.unique(im, return_counts=True)
        num_classes = len(im_u)

        self.palette = im_u.astype(np.uint32)
        self.classes = num_classes

    @tf.function()
    def to_one_hot(self, image):
        image = tf.cast(image, tf.int64)

        x = []
        for i,p in enumerate(self.palette):
            wh = tf.where(image==p, 1, 0)
            wh = tf.cast(wh, dtype=tf.float32)
            wh = tf.reshape(wh, [224, 224])
            x.append(wh)
        output = tf.stack(x, axis=-1)

        return output

    def load_dataset(self, shuffle=True):
        data = [tifffile.imread(data_file).astype(np.float32).reshape([224,224,1]) for data_file in self.data_paths]
        ds = tf.data.Dataset.from_tensor_slices((data, data))
        ds = ds.map(
            self.load_data_label,
            num_parallel_calls=tf.data.AUTOTUNE
            )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.data_paths))
        if self.augmentation:
            ds = ds.map(
                lambda x, y: self.augment((x, y)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds

    @tf.function()
    def augment(self, data_label):
        data, label = data_label

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = tf.image.flip_left_right(data)
            label = tf.image.flip_left_right(label)

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = tf.image.flip_up_down(data)
            label = tf.image.flip_up_down(label)

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            data = self.noise(data)
            label = label

        lot = tf.random.uniform(shape=[1])
        if lot>0.5:
            rotate_time = random.randint(1,3)
            data = tf.image.rot90(data, k=rotate_time)
            label = tf.image.rot90(label, k=rotate_time)

        return data, label

    def noise(self, data):
        no = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=0.2, dtype=tf.float32)
        noise_data = tf.add(data, no)
        return noise_data

def main():
    ds = DatasetTF()
    ds.set_classes()
    x = ds.load_dataset()
    x = iter(x)
    print(next(x))

if __name__ == '__main__':
    main()