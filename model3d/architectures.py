from functools import partial
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, Multiply
from keras.engine import Model
from .unet import create_convolution_block, concatenate


data_format = 'channels_last'

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=False,
                                   data_format=data_format)


def isensee2017_model(input_shape=(128, 128, 128, 1), n_base_filters=16, depth=7, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=4)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), data_format=data_format)(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2), data_format=data_format)(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    return model


def basic_unet(input_shape=(128, 128, 128, 1), n_base_filters=16, depth=7, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, activation_name="sigmoid"):
    inputs = Input(input_shape)
    # y_true = Input(input_shape)
    # heatmap = Input(input_shape)

    current_layer = inputs
    # current_heatmap = heatmap
    level_output_layers = list()
    level_filters = list()
    # heatmap_layers = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
            # base filter = 4
            # heatmap_layer = concatenate([current_heatmap, current_heatmap], axis=4)
            # heatmap_layer = concatenate([heatmap_layer, heatmap_layer], axis=4)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))
            # heatmap_layer = MaxPooling3D(pool_size=(2,2,2), data_format=data_format)(current_heatmap)
            # heatmap_layer = concatenate([heatmap_layer, heatmap_layer], axis=4)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        level_output_layers.append(context_output_layer)
        #heatmap_layers.append(heatmap_layer)
        current_layer = context_output_layer
        #current_heatmap = heatmap_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        # attention_layer = Multiply()([level_output_layers[level_number], heatmap_layers[level_number]])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=4)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), data_format=data_format)(current_layer))

    activation_block = Activation(activation_name)(segmentation_layers[0])
    model = Model(inputs=inputs, outputs=activation_block)
    return model


def vnet(input_shape=(128, 128, 128, 1), n_base_filters=16, depth=7, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, activation_name="sigmoid"):
    inputs = Input(input_shape)
    # y_true = Input(input_shape)
    # heatmap = Input(input_shape)

    current_layer = inputs
    # current_heatmap = heatmap
    level_output_layers = list()
    level_filters = list()
    # heatmap_layers = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
            # base filter = 4
            # heatmap_layer = concatenate([current_heatmap, current_heatmap], axis=4)
            # heatmap_layer = concatenate([heatmap_layer, heatmap_layer], axis=4)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))
            # heatmap_layer = MaxPooling3D(pool_size=(2,2,2), data_format=data_format)(current_heatmap)
            # heatmap_layer = concatenate([heatmap_layer, heatmap_layer], axis=4)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
        # heatmap_layers.append(heatmap_layer)
        # current_heatmap = heatmap_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        # attention_layer = Multiply()([level_output_layers[level_number], heatmap_layers[level_number]])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=4)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), data_format=data_format)(current_layer))

    activation_block = Activation(activation_name)(segmentation_layers[0])
    model = Model(inputs=inputs, outputs=activation_block)
    return model


def no_pooling(input_shape=(128, 128, 128, 1), n_base_filters=16, depth=7, dropout_rate=0.3,
                    n_labels=4, activation_name="sigmoid"):
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        print(n_level_filters)
        level_filters.append(n_level_filters)
        in_conv = create_convolution_block(current_layer, n_level_filters)
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    for level_number in range(depth - 2, -1, -1):
        in_conv = create_convolution_block(current_layer, level_filters[level_number])
        context_output_layer = create_context_module(in_conv, level_filters[level_number], dropout_rate=dropout_rate)
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    output_layer = Conv3D(n_labels, (1, 1, 1), data_format=data_format)(current_layer)
    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size, data_format=data_format)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2