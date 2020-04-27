#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27/4/20 6:09 pm
# @Author  : Jiabo He (Jacobi)
# @Site    : 
# @File    : data_processing.py
# @Software: PyCharm

import numpy as np
from scipy import ndimage


def random_rotation_3d(batch, angle_z, angle_y, angle_x, original):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images and corresponding original images
    """
    size = batch.shape
    batch_rot = np.zeros(size, 'int8')
    batch_original = np.zeros(size, 'int8')
    for i in range(size[0]):
        image1 = batch[i]
        original1 = original[i]
        # rotate along z-axis
        image2 = ndimage.interpolation.rotate(image1, angle_z, order=1, mode='nearest', axes=(0, 1), reshape=True)
        original2 = ndimage.interpolation.rotate(original1, angle_z, order=1, mode='nearest', axes=(0, 1), reshape=True)
        # rotate along y-axis
        image3 = ndimage.interpolation.rotate(image2, angle_y, order=1, mode='nearest', axes=(0, 2), reshape=True)
        original3 = ndimage.interpolation.rotate(original2, angle_y, order=1, mode='nearest', axes=(0, 2), reshape=True)

        # rotate along x-axis
        batch_rot[i] = ndimage.interpolation.rotate(image3, angle_x, order=1, mode='nearest', axes=(1, 2), reshape=True)
        batch_original[i] = ndimage.interpolation.rotate(original3, angle_x, order=1, mode='nearest', axes=(1, 2), reshape=True)

    return batch_rot, batch_original


def mirror_in_xyz(x_train, y_train):
    # mirror in x
    input_mirror1 = np.zeros(x_train.shape, 'int8')
    output_mirror1 = np.zeros(y_train.shape, 'int8')
    for m in range(x_train.shape[0]):
        input1 = np.zeros((64, 64, 64), 'int8')
        output1 = np.zeros((64, 64, 64), 'int8')
        a = x_train[m]
        b = y_train[m]
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    input1[i, j, k] = a[63-i, j, k]
                    output1[i, j, k] = b[63-i, j, k]
        input_mirror1[m] = input1
        output_mirror1[m] = output1
    x_train2 = np.concatenate((x_train, input_mirror1), axis=0)
    y_train2 = np.concatenate((y_train, output_mirror1), axis=0)
    # mirror in y
    input_mirror2 = np.zeros(x_train2.shape, 'int8')
    output_mirror2 = np.zeros(y_train2.shape, 'int8')
    for m in range(x_train2.shape[0]):
        input1 = np.zeros((64, 64, 64), 'int8')
        output1 = np.zeros((64, 64, 64), 'int8')
        a = x_train2[m]
        b = y_train2[m]
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    input1[i, j, k] = a[i, 63-j, k]
                    output1[i, j, k] = b[i, 63-j, k]
        input_mirror2[m] = input1
        output_mirror2[m] = output1
    x_train3 = np.concatenate((x_train2, input_mirror2), axis=0)
    y_train3 = np.concatenate((y_train2, output_mirror2), axis=0)
    # mirror in z
    input_mirror3 = np.zeros(x_train3.shape, 'int8')
    output_mirror3 = np.zeros(y_train3.shape, 'int8')
    for m in range(x_train3.shape[0]):
        input1 = np.zeros((64, 64, 64), 'int8')
        output1 = np.zeros((64, 64, 64), 'int8')
        a = x_train3[m]
        b = y_train3[m]
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    input1[i, j, k] = a[i, j, 63-k]
                    output1[i, j, k] = b[i, j, 63-k]
        input_mirror3[m] = input1
        output_mirror3[m] = output1
    x_train4 = np.concatenate((x_train3, input_mirror3), axis=0)
    y_train4 = np.concatenate((y_train3, output_mirror3), axis=0)
    return x_train4, y_train4


def build_heatmap(y, heatmap):
    for m in range(y.shape[0]):
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                for k in range(y.shape[3]):
                    if y[m,i,j,k] == 1:
                        heatmap[i,j,k] += 1
    return heatmap
