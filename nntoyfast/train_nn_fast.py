# docker run -it -v /scratch:/scratch pteng /bin/bash
# cd /scratch/pteng/sandbox/nn/tubecad
# rm -rf nntoyfast;rm -rf logs/nntoyfast;CUDA_VISIBLE_DEVICES=2 python train_nn_fast.py
# tensorboard --logdir=logs)
# tf 1.14.0, keras 2.2.4
import sys
import traceback
seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)

IMG = 'img'
LUNG = 'lung_mask'
AIRWAY = 'airway_mask'
TRACHEA = 'trachea_mask'
LINE = 'line_mask'
CARINACOORD = 'carina_coord'
STARTCOORD = 'start_cood'
ENDCOORD = 'end_coord'
TUBEORNOT = 'tube_present'
width, height, channel = 512,512,1
LINE_TRUE = 'LINE_TRUE'

import pandas as pd
import numpy as np

import tensorflow as tf
from keras import backend as K

# https://github.com/keras-team/keras/issues/3611
def dice_coef(y_true, y_pred, smooth=1e-10):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth + 1e-10), axis=0)

mean_physical_size = 1.3 #mm
def mm_euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true*255*mean_physical_size - y_pred*255*mean_physical_size), axis=-1))

def dice_coef_loss(y_true, y_pred):
    return tf.reduce_mean(1-dice_coef(y_true, y_pred))

def dice_coef_metric(y_true, y_pred):
    return dice_coef(y_true, y_pred)

def compute_coord_loss(y_true,y_pred):
        loss_px = tf.squared_difference(y_true[...,0],y_pred[...,0])
        loss_py = tf.squared_difference(y_true[...,1],y_pred[...,1])
        coord_loss = tf.add(loss_px,loss_py)    
        return tf.reduce_mean(coord_loss)

def compute_coord_loss_with_weight(y_true_tube):
    def loss(y_true,y_pred):
        loss_px = tf.squared_difference(y_true[...,0],y_pred[...,0])
        loss_py = tf.squared_difference(y_true[...,1],y_pred[...,1])
        coord_loss = tf.add(loss_px,loss_py)    
        coord_loss = tf.multiply(y_true_tube,coord_loss)
        return tf.reduce_mean(coord_loss)
        
    return loss


from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten, Concatenate, LeakyReLU
from keras.layers import Input, GlobalMaxPooling2D
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib.layers import spatial_softmax

import tensorflow as tf

import keras.backend as kb
from keras.layers import Layer

from keras.callbacks import Callback

class MyBuilder:
    droprate=0.2
    def __init__(self):
        pass
    
    def build_shared_layers(self,inputs):
        
        droprate = self.droprate

        def conv_lrelu_bnorm(x,filter_num,size,
            chanDim=-1,padding='same',strides=1):
            x = Conv2D(filter_num, size, padding=padding,strides=strides)(x)
            x = Conv2D(filter_num//2, size, padding=padding,strides=strides)(x)
            x = Conv2D(filter_num, size, padding=padding,strides=strides)(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = LeakyReLU(alpha=0.01)(x)
            x = Dropout(droprate)(x)
            return x

        x = conv_lrelu_bnorm(inputs,16,(3,3))
        x0 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)         

        x = conv_lrelu_bnorm(x,32,(3,3))
        x1 = x
        x = MaxPooling2D(pool_size=(2, 2))(x) 

        x = conv_lrelu_bnorm(x,64,(3,3))
        x2 = x
        x = MaxPooling2D(pool_size=(2, 2))(x) 

        x = conv_lrelu_bnorm(x,128,(3,3))
        x3 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        return x,x0,x1,x2,x3

    def build_segmenter_layers(self,x,x0,x1,x2,x3):
        
        droprate = self.droprate

        def convt_lrelu_bnorm(x,filter_num,size,
            chanDim=-1,padding='same',strides=1):
            x = Conv2DTranspose(filter_num, size,strides=strides,padding=padding)(x)
            x = Conv2DTranspose(filter_num//2, size,strides=strides,padding=padding)(x)
            x = Conv2DTranspose(filter_num, size,strides=strides,padding=padding)(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = LeakyReLU(alpha=0.01)(x)
            x = Dropout(droprate)(x)
            return x

        x = convt_lrelu_bnorm(x,128,(3,3))
        x = UpSampling2D(size=(2,2),)(x)

        x = Concatenate(axis=-1)([x,x3])
        x = convt_lrelu_bnorm(x,64,(3,3))
        x = UpSampling2D(size=(2,2),)(x)

        x = Concatenate(axis=-1)([x,x2])
        x = convt_lrelu_bnorm(x,32,(3,3))
        x = UpSampling2D(size=(2,2),)(x)

        x = Concatenate(axis=-1)([x,x1])
        x = convt_lrelu_bnorm(x,16,(3,3))
        x = UpSampling2D(size=(2,2),)(x)
        
        x = Concatenate(axis=-1)([x,x0])
        f = Conv2D(1, (3, 3), padding='same')(x)
        f = Activation("sigmoid",name=LINE)(f)
        return f


    def build_tube_layers(self,x):
        
        droprate = self.droprate

        def lrelu_bnorm(x,filter_num):
            x = Dense(filter_num)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.01)(x)
            x = Dropout(droprate)(x)
            return x

        x = Flatten()(x)

        x = lrelu_bnorm(x,1024)
        x = lrelu_bnorm(x,128)
        x = lrelu_bnorm(x,64)
        x = lrelu_bnorm(x,64)
        x = lrelu_bnorm(x,32)
        
        tube = Dense(1)(x)
        tube = Activation("sigmoid",name=TUBEORNOT)(tube)
        
        return tube

    def build(self,width, height, channel,stop_gradient_prior_coord=False):        
        
        K.clear_session()             
        K.set_learning_phase(1) # important for image tensorboard callback

        self.stop_gradient_prior_coord = stop_gradient_prior_coord
        
        inputShape = (height, width, channel)
        chanDim = -1
        
        self.inputs = Input(shape=inputShape,name=IMG)
        
        self.line_true = Input(shape=inputShape,name=LINE_TRUE)
        
        self.shared_output,x0,x1,x2,x3 = self.build_shared_layers(self.inputs)
        self.line= self.build_segmenter_layers(self.shared_output,x0,x1,x2,x3)
        self.tube = self.build_tube_layers(self.shared_output)
        self.model = Model(
            inputs=[self.inputs,self.line_true],
            outputs=[self.tube,self.line],
            name="cad")

        return self.model


if __name__ == "__main__":
    
    exper_name = 'nntoyfast'
    
    batch_size = 8
    builder = MyBuilder()
    model = builder.build(width, height, channel,stop_gradient_prior_coord=False)
    print(model.summary())
    init_lr = 1e-3
    opt = Adam(lr=init_lr)

    metrics = {
        LINE: dice_coef_metric,
        TUBEORNOT: "accuracy",
    }

    losses = {
        LINE: dice_coef_loss,
        TUBEORNOT: "binary_crossentropy",
    }

    lossWeights = {
        LINE: 1,
        TUBEORNOT: 1,
    }    

    model.compile(optimizer=opt, loss=losses,
        loss_weights=lossWeights,metrics=metrics)

    model_yml_path = exper_name+'.yml'
    with open(model_yml_path,'w') as f:
        f.write(model.to_yaml())

    with open(model_yml_path,'r') as f:
        yaml_string = f.read()
    
    from keras.models import model_from_yaml
    model = model_from_yaml(yaml_string)
    print(model)