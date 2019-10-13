import keras.layers
import keras.models
from keras.layers import BatchNormalization, Conv2D
import tensorflow as tf
from keras import backend as K

CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"mode": 0, "momentum": 0.9}


# returns a core model from gray input to 64 channels of the same size
def get_core_3(dim1, dim2, input_channel=1):
    x = keras.layers.Input(shape=(dim1, dim2, input_channel))

    a = Conv2D(64, (3, 3), **option_dict_conv)(x)
    a = BatchNormalization(**option_dict_bn)(a)
    a = Conv2D(64, (3, 3), **option_dict_conv)(a)
    a = BatchNormalization(**option_dict_bn)(a)
    y = keras.layers.MaxPooling2D()(a)

    b = Conv2D(128, (3, 3), **option_dict_conv)(y)
    b = BatchNormalization(**option_dict_bn)(b)
    b = Conv2D(128, (3, 3), **option_dict_conv)(b)
    b = BatchNormalization(**option_dict_bn)(b)
    y = keras.layers.MaxPooling2D()(b)

    c = Conv2D(256, (3, 3), **option_dict_conv)(y)
    c = BatchNormalization(**option_dict_bn)(c)
    c = Conv2D(256, (3, 3), **option_dict_conv)(c)
    c = BatchNormalization(**option_dict_bn)(c)
    y = keras.layers.MaxPooling2D()(c)

    d = Conv2D(512, (3, 3), **option_dict_conv)(y)
    d = BatchNormalization(**option_dict_bn)(d)
    d = Conv2D(512, (3, 3), **option_dict_conv)(d)
    d = BatchNormalization(**option_dict_bn)(d)

    # UP

    d = keras.layers.UpSampling2D()(d)
    y = keras.layers.merge.concatenate([d, c], axis=3)

    e = Conv2D(256, (3, 3), **option_dict_conv)(y)
    e = BatchNormalization(**option_dict_bn)(e)
    e = Conv2D(256, (3, 3), **option_dict_conv)(e)
    e = BatchNormalization(**option_dict_bn)(e)
    e = keras.layers.UpSampling2D()(e)
    y = keras.layers.merge.concatenate([e, b], axis=3)

    f = Conv2D(128, (3, 3), **option_dict_conv)(y)
    f = BatchNormalization(**option_dict_bn)(f)
    f = Conv2D(128, (3, 3), **option_dict_conv)(f)
    f = BatchNormalization(**option_dict_bn)(f)
    f = keras.layers.UpSampling2D()(f)
    y = keras.layers.merge.concatenate([f, a], axis=3)

    y = Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = BatchNormalization(**option_dict_bn)(y)
    y = Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = BatchNormalization(**option_dict_bn)(y)

    return [x, y]

def get_core_2(dim1, dim2, input_channel=1):
    x = keras.layers.Input(shape=(dim1, dim2, input_channel))

    a = Conv2D(64, (3, 3), **option_dict_conv)(x)
    a = BatchNormalization(**option_dict_bn)(a)
    a = Conv2D(64, (3, 3), **option_dict_conv)(a)
    a = BatchNormalization(**option_dict_bn)(a)
    y = keras.layers.MaxPooling2D()(a)

    b = Conv2D(128, (3, 3), **option_dict_conv)(y)
    b = BatchNormalization(**option_dict_bn)(b)
    b = Conv2D(128, (3, 3), **option_dict_conv)(b)
    b = BatchNormalization(**option_dict_bn)(b)
    y = keras.layers.MaxPooling2D()(b)

    c = Conv2D(256, (3, 3), **option_dict_conv)(y)
    c = BatchNormalization(**option_dict_bn)(c)
    c = Conv2D(256, (3, 3), **option_dict_conv)(c)
    c = BatchNormalization(**option_dict_bn)(c)

    # UP

    c = keras.layers.UpSampling2D()(c)
    y = keras.layers.merge.concatenate([b, c], axis=3)

    e = Conv2D(128, (3, 3), **option_dict_conv)(y)
    e = BatchNormalization(**option_dict_bn)(e)
    e = Conv2D(128, (3, 3), **option_dict_conv)(e)
    e = BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)
    y = keras.layers.merge.concatenate([e, a], axis=3)

    y = Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = BatchNormalization(**option_dict_bn)(y)
    y = Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = BatchNormalization(**option_dict_bn)(y)

    return [x, y]


def get_model(dim1, dim2, activation=None, 
                    input_channel=1, output_channel=3):
    [x, y] = get_core_2(dim1, dim2, input_channel)
    if activation is not None:
        y = Conv2D(output_channel, (1, 1), activation = activation, padding = "same")(y)
    else:
        y = Conv2D(output_channel, (1, 1), **option_dict_conv)(y)
    # since in objectives.py use softmax_cross_entropy
    # here should not use softmax activation but just logit 
    # y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model
