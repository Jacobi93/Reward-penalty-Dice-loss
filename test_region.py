#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 29/7/19 5:56 pm
# @Author  : Jiabo He (Jacobi)
# @Site    : 
# @File    : test_region.py
# @Software: PyCharm


import numpy as np
import tensorflow as tf

def jaccard_and_dice(surgeon1, surgeon2):
    surgeon1 = surgeon1.flatten()
    surgeon2 = surgeon2.flatten()
    intersection = np.sum(surgeon1 * surgeon2)
    jaccard = intersection/(np.sum(surgeon1) + np.sum(surgeon2) - intersection)
    dice = 2 * intersection / (np.sum(surgeon1) + np.sum(surgeon2))
    return jaccard, dice

def build_heatmap(y, heatmap):
    for m in range(y.shape[0]):
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                for k in range(y.shape[3]):
                    if y[m,i,j,k] == 1:
                        heatmap[i,j,k] += 1
    return heatmap

def rp_dice(surgeon1, surgeon2, heatmap):
    surgeon1 = surgeon1.flatten()
    surgeon2 = surgeon2.flatten()
    heatmap = heatmap.flatten()
    intersection = np.sum(surgeon1 * surgeon2 * heatmap)
    dice = 2 * intersection / (np.sum(surgeon1 * np.abs(heatmap)) + np.sum(surgeon2 * np.abs(heatmap)))
    return dice


# compare different training surgeons with models
x9_7 = np.load('cm_data/x9_7.npy')
y9_7 = np.load('cm_data/y9_7.npy')
num_surgeon = 7

for s in [1,2,3,4,5,6,7]:
    model = tf.keras.models.load_model('model_surgeon/unet_s{}_dice.h5'.format(s), compile=False)
    surgeon = s
    print('surgeon:', surgeon)
    x_train = np.delete(x9_7, [surgeon-1, surgeon-1+num_surgeon, surgeon-1+2*num_surgeon, surgeon-1+3*num_surgeon, surgeon-1+4*num_surgeon,
                   surgeon-1+5*num_surgeon, surgeon-1+6*num_surgeon, surgeon-1+7*num_surgeon, surgeon-1+8*num_surgeon], axis=0)
    y_train = np.delete(y9_7, [surgeon-1, surgeon-1+num_surgeon, surgeon-1+2*num_surgeon, surgeon-1+3*num_surgeon, surgeon-1+4*num_surgeon,
                   surgeon-1+5*num_surgeon, surgeon-1+6*num_surgeon, surgeon-1+7*num_surgeon, surgeon-1+8*num_surgeon], axis=0)
    x_test = x9_7[
             [surgeon - 1, surgeon - 1 + num_surgeon, surgeon - 1 + 2 * num_surgeon, surgeon - 1 + 3 * num_surgeon,
              surgeon - 1 + 4 * num_surgeon,
              surgeon - 1 + 5 * num_surgeon, surgeon - 1 + 6 * num_surgeon, surgeon - 1 + 7 * num_surgeon,
              surgeon - 1 + 8 * num_surgeon], :]
    y_test = y9_7[
             [surgeon - 1, surgeon - 1 + num_surgeon, surgeon - 1 + 2 * num_surgeon, surgeon - 1 + 3 * num_surgeon,
              surgeon - 1 + 4 * num_surgeon,
              surgeon - 1 + 5 * num_surgeon, surgeon - 1 + 6 * num_surgeon, surgeon - 1 + 7 * num_surgeon,
              surgeon - 1 + 8 * num_surgeon], :]

    x = np.expand_dims(x_test/6.5-1, axis=5)
    right = model.predict(x)
    right = np.squeeze(right)

    # generate [0, 1] output
    output_01 = np.zeros(right.shape, 'int8')
    threshold = 0.5
    for m in range(right.shape[0]):
        for i in range(right.shape[1]):
            for j in range(right.shape[2]):
                for k in range(right.shape[3]):
                    if right[m,i,j,k] > threshold:
                        output_01[m,i,j,k] = 1
    unique, counts = np.unique(output_01, return_counts=True)
    print(dict(zip(unique, counts)))
    print(round(counts[1]/right.shape[0]))
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))
    print(round(counts[1]/right.shape[0]))

    # compare surgeons with dice
    dice = np.zeros((output_01.shape[0], 6))
    for i in range(output_01.shape[0]):
        for j in range(6):
            surgeon1 = output_01[i] # output of model
            surgeon2 = y_train[6*i+j]
            _, dice[i,j] = jaccard_and_dice(surgeon1, surgeon2)

    mean_dice = np.mean(dice)
    std_dice = np.std(dice)
    print(round(mean_dice,4), round(std_dice,4))

    # compare surgeons with heatmap dice
    penalty = -6
    heatmap_single = np.zeros((int(y_train.shape[0] / (num_surgeon - 1)), 64, 64, 64), 'int8')
    for i in range(heatmap_single.shape[0]):
        heatmap_single[i] = build_heatmap(y_train[i * (num_surgeon - 1):(i + 1) * (num_surgeon - 1)], heatmap_single[i])
    heatmap_single[heatmap_single == 0] = penalty
    heat_dice = np.zeros((output_01.shape[0], 6))

    dice = np.zeros((output_01.shape[0], 6))
    for i in range(output_01.shape[0]):
        for j in range(6):
            surgeon1 = output_01[i] # output of model
            surgeon2 = y_train[6*i+j]
            heat = heatmap_single[i]
            heat_dice[i,j] = rp_dice(surgeon1, surgeon2, heat)
    mean_dice = np.mean(heat_dice)
    std_dice = np.std(heat_dice)
    print(round(mean_dice,4), round(std_dice,4))


# test on different bones
x9_7 = np.load('cm_data/x9_7.npy')
y9_7 = np.load('cm_data/y9_7.npy')
num_surgeon = 7
for b in [1,2,3,4,5,6,7,8,9]:
    model = tf.keras.models.load_model('model_bone/unet_bone{}_heatmapdice.h5'.format(b), compile=False)
    bone = b
    print('bone:', bone)
    x_test = x9_7[(bone-1)*num_surgeon:bone*num_surgeon ,:]
    y_test = y9_7[(bone-1)*num_surgeon:bone*num_surgeon ,:]

    # predict separately due to memory limit
    x = np.expand_dims(x_test/6.5-1, axis=5)
    right = model.predict(x)
    right = np.squeeze(right)

    # generate [0, 1] output
    output_01 = np.zeros(right.shape, 'int8')
    threshold = 0.5
    for m in range(right.shape[0]):
        for i in range(right.shape[1]):
            for j in range(right.shape[2]):
                for k in range(right.shape[3]):
                    if right[m,i,j,k] > threshold:
                        output_01[m,i,j,k] = 1
    unique, counts = np.unique(output_01, return_counts=True)
    print(dict(zip(unique, counts)))
    print(round(counts[1]/right.shape[0]))
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))
    print(round(counts[1]/right.shape[0]))

    # Jaccard and DICE similarity in the down scale
    # dice
    jaccard = np.zeros(right.shape[0])
    dice = np.zeros(right.shape[0])

    for j in range(right.shape[0]):
        surgeon1 = output_01[j] # output of model
        surgeon2 = y_test[j]
        jaccard[j], dice[j] = jaccard_and_dice(surgeon1, surgeon2)

    mean_dice = np.mean(dice)
    std_dice = np.std(dice)
    mean_jaccard = np.mean(jaccard)
    std_jaccard = np.std(jaccard)
    print(round(mean_dice, 4), round(std_dice, 4))

    # with heatmap dice
    penalty = -7
    heatmap = np.zeros((64, 64, 64), 'int8')
    heatmap = build_heatmap(y_test, heatmap)
    heatmap[heatmap == 0] = penalty
    heat_dice = np.zeros(right.shape[0])

    for j in range(right.shape[0]):
        surgeon1 = output_01[j] # output of model
        surgeon2 = y_test[j]
        heat_dice[j] = rp_dice(surgeon1, surgeon2, heatmap)

    mean_dice = np.mean(heat_dice)
    std_dice = np.std(heat_dice)
    print(round(mean_dice,4), round(std_dice,4))