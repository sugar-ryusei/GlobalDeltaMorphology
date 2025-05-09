import numpy as np
import pandas as pd
import os
import shutil
import random
import glob
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from ConvAE import CAE
import dataset


def train(
    model_name,
    epochs=200,
    batch_size=8,
    num_feature=20,
    loss='mse',
    opt='adam',
    augmentation=True,
    input_shape=(28, 28, 1),#
    data_path='',
    save_direc='',
    result_exist=False,
    callback_period=50,
    lr=0.001,
    decay=1e-6,
    tlmodel_path=''
    ):
    
    print('prepare dataset')
    image_files = glob.glob(os.path.join(data_path, "*.tiff"))
    random.shuffle(image_files)
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    gen_train = dataset.DatasetTF(                
                batch_size=batch_size,
                augmentation=augmentation,
                data_path=train_files
                )
    gen_train.set_classes()
    ds_train = gen_train.load_dataset(shuffle=True)
    
    gen_test = dataset.DatasetTF(
                batch_size=batch_size,
                augmentation=False,
                data_path=test_files)
    gen_test.palette = gen_train.palette
    gen_test.classes = gen_train.classes
    ds_test = gen_test.load_dataset(shuffle=False)
    
    print('building model')
    model = CAE(input_shape=input_shape, classes=gen_train.classes, num_feature=num_feature, tlmodel_path=tlmodel_path)

    if loss == 'CE':
        loss_func = [tf.keras.losses.CategoricalCrossentropy()]

    if loss == 'mse':
        loss_func = 'mse'

    if opt == 'adam':
        opt = tf.keras.optimizers.Adam(
            lr=lr,
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-07,
            decay=0.0,
            amsgrad=False)
    
    if opt == 'sgd':
        opt = tf.keras.optimizers.SGD(
            lr=lr,
            decay=decay)

    if save_direc != '':
        save_direc = save_direc + os.sep
    else:
        save_direc = './'
    os.makedirs(save_direc, exist_ok=result_exist)
    log_filepath = os.path.join(save_direc + 'logs_' + model_name)
    model_out = save_direc + model_name + '.h5'
    model_out_e = save_direc + model_name + '_e{epoch:04d}.h5'
    
    #callback
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
    callback_period = int(np.ceil(gen_train.data_length / batch_size) * callback_period)
    me_cb = tf.keras.callbacks.ModelCheckpoint(
        model_out_e, monitor='val_loss', verpose=0, save_best_only=False,
        save_wight_only=False, save_freq=callback_period)
    
    callbacks = [tb_cb, me_cb]
    
    model.compile(loss=loss_func, optimizer=opt)
    model.summary()

    print('training start')

    
    history = model.fit(
        x=ds_train,
        epochs=epochs,
        verbose=1,
        validation_data=ds_test,
        shuffle=True,
        callbacks=callbacks,
        use_multiprocessing=True
        )

    df_his = pd.DataFrame(history.history)
    df_his.to_csv(model_out.replace('.h5', '.csv'))

    model.save(model_out)

def main():
    train(
    model_name='dcec',
    epochs=100,
    batch_size=8,
    num_feature=70,
    loss='CE',
    opt='adam',
    augmentation=True,
    input_shape=(224, 224, 1),#
    data_path='delta_image',
    save_direc='history/1',
    result_exist=True,
    callback_period=20,
    lr=0.00011,
    decay=1e-6,
    tlmodel_path=''
    )

if __name__ == '__main__':
    main()