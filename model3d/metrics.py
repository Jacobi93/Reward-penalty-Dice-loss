import tensorflow as tf
from functools import partial
from keras import backend as K

# determine the weight based on the overall ratio of background over foreground in the dataset
def weighted_binary_crossentropy(y_true, y_pred, smooth=1e-6, weight_1=23.485):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    bce = weight_1 * K.sum(y_true * K.log(y_pred))
    bce += K.sum((1 - y_true) * K.log(1 - y_pred))
    return -bce


def dice_coefficient(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def dynamic_weighted_dice(y_true, y_pred, heatmap, smooth=1.):
    intersection = K.sum(y_true * y_pred * heatmap)
    return (2. * intersection + smooth) / (K.sum(y_true * K.abs(heatmap)) + K.sum(y_pred * K.abs(heatmap)) + smooth)


def dynamic_weighted_dice_loss(y_true, y_pred, heatmap):
    return 1 - dynamic_weighted_dice(y_true, y_pred, heatmap)


# the loss function is adjusted when programming so that there are only two inputs for the loss
# it is equivalent to the rp dice loss function in the paper when training
def rp_dice(heatmap, y_pred, smooth=1.):
    intersection = K.sum(heatmap * y_pred)
    return (2. * intersection + smooth) / (K.sum(K.abs(heatmap)) + K.sum(y_pred) + smooth)


def rp_dice_loss(heatmap, y_pred):
    return 1 - rp_dice(heatmap, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f