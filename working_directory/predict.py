import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import time
import cv2
import os
import glob
import tensorflow as tf
import tifffile


class PredictTest(object):

    def __init__(self,
                image_size,
                batch_size,
                test_path,
                index,
                model_name
                ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_path = sorted(glob.glob(f'{test_path}/*.tiff'))
        self.index = index
        self.model_name = model_name
        self.pred_path = f"predict/{self.index}/"
        self.feature_path = f"feature/{self.index}/"
        self.model_path = f"history/{self.index}/{self.model_name}.h5"
        self.data_names = [os.path.basename(path) for path in self.test_path]
    

    def load_dataset(self):
        data = [tifffile.imread(data_file).astype(np.float32).reshape([224,224,1]) for data_file in self.test_path]
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.batch(self.batch_size)
        return ds

    def predict(self, data):
        model = tf.keras.models.load_model(self.model_path)
        hidden = model.get_layer(name='embedding').output
        model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, hidden])
        pred, features = model.predict(data)
        return pred, features

    def conv(self, pred):
        images = pred[:,:,:,1].astype(np.float32)
        return images

    def save_images(self, images):
        os.makedirs(self.pred_path, exist_ok=True)
        for i in range(len(images)):
            temp_path = f'{self.pred_path}/{self.data_names[i]}'
            tifffile.imsave(temp_path, images[i])

    def save_features(self, features):
        os.makedirs(self.feature_path, exist_ok=True)
        data_names = [os.path.splitext(path)[0] for path in self.data_names]
        df = pd.DataFrame(features, index=data_names)
        df.to_csv(f'{self.feature_path}{self.index}.csv')

    def conduct_predict(self):
        data = self.load_dataset()
        pred, features = self.predict(data)
        images = self.conv(pred)
        self.save_images(images)
        self.save_features(features)

def main():
    pt = PredictTest(image_size=224,
                     batch_size=8,
                     test_path = "delta_image",
                     index = 1,
                     model_name = "dcec")
    pt.conduct_predict()

if __name__ == '__main__':
    main()