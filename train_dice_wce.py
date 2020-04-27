#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/9/19 1:41 pm
# @Author  : Jiabo He (Jacobi)
# @Site    : 
# @File    : train_dice.py
# @Software: PyCharm


import os, argparse
import numpy as np
import tensorflow as tf
from model3d.architectures import isensee2017_model, basic_unet, vnet
from data_processing import random_rotation_3d, mirror_in_xyz
from keras.optimizers import Adam
from model3d.metrics import weighted_binary_crossentropy, dice_coefficient_loss
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from tqdm import tqdm


# In experiment 1, different bones are picked for testing
# In experiment 2, different surgeons are picked for testing
parser = argparse.ArgumentParser()
# parser.add_argument('--surgeon', type=int, default=1)
parser.add_argument('--bone', type=int, default=1)
args = parser.parse_args()

config = dict()
# config["surgeon"] = args.surgeon
# print('surgeon: {}'.format(config["surgeon"]))
config["bone"] = args.bone
print('bone: {}'.format(config["bone"]))
config["image_shape"] = (64, 64, 64)  # This determines what shape the images will be cropped/resampled to.
config["n_base_filters"] = 4
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["initial_learning_rate"] = 1e-4
config["batch_size"] = 64
config["n_labels"] = 1
config["nb_channels"] = 1
config["input_shape"] = tuple(list(config["image_shape"]) + [config["nb_channels"]])
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution


def main():
    if os.path.exists('model_bone/unet_bone{}_dice.h5'.format(config["bone"])):
        print('Import the pre-trained model!')
        model = tf.keras.models.load_model('model_bone/unet_bone{}_dice.h5'.format(config["bone"]),
                                           compile=False,
                                           custom_objects={'dice_coefficient_loss': dice_coefficient_loss})
    else:
        model = basic_unet(input_shape=config["input_shape"], n_labels=config["n_labels"], depth=5,
                                  dropout_rate=0.3, n_base_filters=config["n_base_filters"])

    # train for different surgeons
    # s1 = config["surgeon"] - 1
    # x9_7 = np.load('cm_data/x9_7.npy')
    # y9_7 = np.load('cm_data/y9_7.npy')
    # num_surgeon = 7
    # x9_7 = np.delete(x9_7, [s1, s1+num_surgeon, s1+2*num_surgeon, s1+3*num_surgeon, s1+4*num_surgeon,
    #                         s1+5*num_surgeon, s1+6*num_surgeon, s1+7*num_surgeon, s1+8*num_surgeon], axis=0)
    # y9_7 = np.delete(y9_7, [s1, s1+num_surgeon, s1+2*num_surgeon, s1+3*num_surgeon, s1+4*num_surgeon,
    #                         s1+5*num_surgeon, s1+6*num_surgeon, s1+7*num_surgeon, s1+8*num_surgeon], axis=0)

    # train for different bones
    b1 = config["bone"] - 1
    x9_7 = np.load('cm_data/x9_7.npy')
    y9_7 = np.load('cm_data/y9_7.npy')
    num_surgeon = 7
    x9_7 = np.delete(x9_7, [b1*num_surgeon, b1*num_surgeon+1, b1*num_surgeon+2, b1*num_surgeon+3,
                            b1*num_surgeon+4, b1*num_surgeon+5, b1*num_surgeon+6], axis=0)
    y9_7 = np.delete(y9_7, [b1*num_surgeon, b1*num_surgeon+1, b1*num_surgeon+2, b1*num_surgeon+3,
                            b1*num_surgeon+4, b1*num_surgeon+5, b1*num_surgeon+6], axis=0)

    # mirror in x y and z axes
    x_mirror, y_mirror = mirror_in_xyz(x9_7, y9_7)

    # rotate in x y and z axes
    x = np.zeros((27, x_mirror.shape[0], 64, 64, 64), 'int8')
    y = np.zeros((27, y_mirror.shape[0], 64, 64, 64), 'int8')
    m = -1
    rotation_angle = [90, 180, 270]
    for i in rotation_angle:
        for j in tqdm(rotation_angle):
            for k in rotation_angle:
                m += 1
                y[m], x[m] = random_rotation_3d(y_mirror, i, j, k, x_mirror)
    x = np.reshape(x, (27 * x_mirror.shape[0], 64, 64, 64))
    y = np.reshape(y, (27 * y_mirror.shape[0], 64, 64, 64))
    # normalization
    x = np.array(x/6.5-1, 'float32')
    x = np.expand_dims(x, 4)
    y = np.expand_dims(y, 4)
    print(np.unique(x), x.shape)
    print(np.unique(y), y.shape)
    print(model.summary())

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coefficient_loss)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    csv_logger = CSVLogger('model/unet_bone{}_dice.csv'.format(config["bone"]),
                           separator=',', append=False)
    mc = ModelCheckpoint('model/unet_bone{}_dice.h5'.format(config["bone"]),
                    monitor='val_loss', mode='auto', save_best_only=True, verbose=1)
    model.fit(x, y, batch_size=config["batch_size"], epochs=config["n_epochs"],
              validation_split=0.1, callbacks=[rlrop, es, csv_logger, mc], shuffle=True)

if __name__ == "__main__":
    main()