
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
from keras import backend as K

from keras.models import Sequential, load_model
from keras import optimizers as opt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, Conv2DTranspose, UpSampling2D
from keras.callbacks import LearningRateScheduler,EarlyStopping

from keras.models import Model
from keras.layers import Permute, Reshape, Lambda, add, Input, Concatenate
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras import regularizers, initializers




epsilon = 1E-8
def compute_obj_loss(y_true,y_pred):
    return tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true[...,-1], y_pred[...,-1]))

def compute_coord_loss(y_true,y_pred):
    loss_px = tf.squared_difference(y_true[...,0],y_pred[...,0])
    loss_py = tf.squared_difference(y_true[...,1],y_pred[...,1])
    loss_wx = tf.squared_difference(y_true[...,2],y_pred[...,2])
    loss_wy = tf.squared_difference(y_true[...,3],y_pred[...,3])
    
    loss_pos = tf.add(loss_px,loss_py)
    loss_width = tf.add(loss_wx,loss_wy)
    
    coord_loss = tf.add(loss_pos,loss_width)
    coord_loss = tf.multiply(y_true[...,-1],coord_loss)
    return tf.reduce_mean(coord_loss)

def compute_iou_loss(y_true,y_pred):
    # ref. https://github.com/ksanjeevan/dourflow/blob/master/net/netloss.py
    def process_boxes(A):
        # ALign x-w, y-h
        A_xy = A[..., 0:2]
        A_wh = A[..., 2:4]
        
        A_wh_half = A_wh / 2.
        # Get x_min, y_min
        A_mins = A_xy - A_wh_half
        # Get x_max, y_max
        A_maxes = A_xy + A_wh_half
        
        return A_mins, A_maxes, A_wh
    
    # Process two sets
    A2_mins, A2_maxes, A2_wh = process_boxes(y_pred)
    A1_mins, A1_maxes, A1_wh = process_boxes(y_true)
    
    # Intersection as min(Upper1, Upper2) - max(Lower1, Lower2)
    intersect_mins  = K.maximum(A2_mins,  A1_mins)
    intersect_maxes = K.minimum(A2_maxes, A1_maxes)
    
    # Getting the intersections in the xy (aka the width, height intersection)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)

    # Multiply to get intersecting area
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Values for the single sets
    true_areas = A1_wh[..., 0] * A1_wh[..., 1]
    pred_areas = A2_wh[..., 0] * A2_wh[..., 1]
    
    # Compute union for the IoU
    union_areas = pred_areas + true_areas - intersect_areas
    
    # probably not necssary...
    iou_loss = tf.multiply(y_true[...,-1], 1. - tf.truediv(intersect_areas,(union_areas+epsilon)) )
    
    return tf.reduce_mean(iou_loss)

def vec_loss(y_true, y_pred):
    
    lambda_obj = 1.0
    lambda_coord = 0.1
    lambda_iou = 1.0
    
    obj_loss = compute_obj_loss(y_true, y_pred)
    obj_loss = tf.multiply(lambda_obj,obj_loss)
    
    coord_loss = compute_coord_loss(y_true, y_pred)
    coord_loss = tf.multiply(lambda_coord,coord_loss)
    
    iou_loss = compute_iou_loss(y_true, y_pred)
    iou_loss = tf.multiply(lambda_iou,iou_loss)
    
    total_loss = tf.add(tf.add(obj_loss,coord_loss),iou_loss)
    
    return tf.reduce_mean(total_loss)

# https://github.com/keras-team/keras/issues/3611
def dice_coef(y_true, y_pred, smooth=1e-8):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return tf.truediv( 2. * intersection + smooth, union + smooth)

def dice_coef_loss(y_true, y_pred):    
    dice = 1-dice_coef(y_true, y_pred)
    # get rid of dice where there is no object.
    is_obj = K.sum(y_true, axis=[1,2])
    
    #dice = dice[...,-1]
    #is_obj = is_obj[...,-1]
    is_obj = K.greater(is_obj,0.5*K.ones_like(is_obj))

    dice = tf.boolean_mask(dice, is_obj)
    return tf.reduce_mean(dice)

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    # https://github.com/guigzzz/Keras-Yolo-v2/blob/f61286371cdc2d470e0811234f552c70bbd5caba/yolo_layer_utils.py#L18
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

def convt_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2DTranspose(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

MASK = 'MASK'
VECTOR = 'VECTOR'

def get_mask_yolo_model(szx,szy):
    inputs = Input(shape=(szx,szy,1))

    dropoutrate = 0.4

    # down sample
    xd = conv_batch_lrelu(inputs, 16, 3)
    xd = MaxPooling2D(pool_size=(2, 2),strides=2)(xd)
    xd=Dropout(dropoutrate)(xd)
    
    xd = conv_batch_lrelu(xd, 32, 3)
    xd = MaxPooling2D(pool_size=(2, 2),strides=2)(xd)
    xd=Dropout(dropoutrate)(xd)

    xd = conv_batch_lrelu(xd, 64, 3)
    xd = conv_batch_lrelu(xd, 32, 1)
    xd = conv_batch_lrelu(xd, 64, 3)
    xd = MaxPooling2D(pool_size=(2, 2),strides=2)(xd)
    xd=Dropout(dropoutrate)(xd)

    xd = conv_batch_lrelu(xd, 128, 3)
    xd = conv_batch_lrelu(xd, 64, 1)
    xd = conv_batch_lrelu(xd, 128, 3)
    xd = MaxPooling2D(pool_size=(2, 2),strides=2)(xd)
    xd=Dropout(dropoutrate)(xd)

    xd = conv_batch_lrelu(xd, 128, 3)
    xd = conv_batch_lrelu(xd, 64, 1)
    xd = conv_batch_lrelu(xd, 128, 3)
    xd = MaxPooling2D(pool_size=(2, 2),strides=2)(xd)
    xd=Dropout(dropoutrate)(xd)

    # to coord & obj
    xb = conv_batch_lrelu(xd, 128, 3)
    xb = conv_batch_lrelu(xb, 64, 1)
    xb = conv_batch_lrelu(xb, 128, 3)
    xb = Dropout(dropoutrate)(xb)
    xb = conv_batch_lrelu(xb, 64, 1)
    xb = conv_batch_lrelu(xb, 128, 3)
    xb = Conv2D(5, (1, 1), strides=1, padding='same',use_bias=False)(xb)
    vecs=Activation('sigmoid',name=VECTOR)(xb)

    # up sample
    merged = Concatenate(axis=-1,)([xd,vecs])
    xu = convt_batch_lrelu(merged, 128, 3) # can just use conv_batch_lrelu if using upsampling2d...
    xu = convt_batch_lrelu(xu, 64, 1)
    xu = convt_batch_lrelu(xu, 128, 3)
    xu = UpSampling2D(size=(2,2),interpolation='nearest')(xu)
    xu = convt_batch_lrelu(xu, 64, 3)
    xu = convt_batch_lrelu(xu, 32, 1)
    xu = convt_batch_lrelu(xu, 64, 3)
    xu = UpSampling2D(size=(2,2),interpolation='nearest')(xu)
    xu = convt_batch_lrelu(xu, 32, 3)
    xu = UpSampling2D(size=(2,2),interpolation='nearest')(xu)
    xu = convt_batch_lrelu(xu, 32, 3)
    xu = UpSampling2D(size=(2,2),interpolation='nearest')(xu)
    xu = convt_batch_lrelu(xu, 16, 3)
    xu = UpSampling2D(size=(2,2),interpolation='nearest')(xu)

    xu = Conv2D(64, (1, 1), strides=1, padding='same',use_bias=False)(xu)
    masks=Activation('sigmoid',name=MASK)(xu)

    # merge outputs
    model = Model(inputs=inputs, outputs=[vecs,masks])
    return model