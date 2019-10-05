import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from make_line import make_data, get_grid

szx,szy,szz=64,64,64
smx,smy=8,8
grid, grid_anchor = get_grid()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

SEED = 0
import os
import random as rn
import numpy as np
from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
set_random_seed(SEED)
rn.seed(SEED)

from keras.models import Sequential, load_model
from keras import optimizers as opt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, Conv2DTranspose
from keras.callbacks import LearningRateScheduler,EarlyStopping

from keras.models import Model
from keras.layers import Permute, Reshape, Lambda, add, Input, Concatenate
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras import regularizers, initializers


from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from hyperopt.hp import choice
from hyperopt.hp import randint
from hyperopt.hp import pchoice
from hyperopt.hp import uniform
from hyperopt.hp import quniform
from hyperopt.hp import loguniform
from hyperopt.hp import qloguniform
from hyperopt.hp import normal
from hyperopt.hp import qnormal
from hyperopt.hp import lognormal
from hyperopt.hp import qlognormal

def create_model(x_train, y_train, x_test, y_test):
    szx,szy=64,64
    smx,smy=8,8
    epsilon = 1E-8
    def compute_obj_loss(y_true,y_pred):
        return tf.keras.backend.binary_crossentropy(y_true[...,-1], y_pred[...,-1])

    def compute_coord_loss(y_true,y_pred):
        loss_px = tf.squared_difference(y_true[...,0], y_pred[...,0])
        loss_py = tf.squared_difference(y_true[...,1], y_pred[...,1])
        loss_wx = tf.squared_difference(y_true[...,2], y_pred[...,2])
        loss_wy = tf.squared_difference(y_true[...,3], y_pred[...,3])

        loss_pos = tf.add(loss_px,loss_py)
        loss_width = tf.add(loss_wx,loss_wy)

        coord_loss = tf.add(loss_pos,loss_width)
        coord_loss = tf.multiply(y_true[...,-1],coord_loss)
        return coord_loss

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
        iou_loss = tf.multiply(y_true[...,-1], tf.truediv(intersect_areas,(union_areas+epsilon)) )

        return iou_loss


    def vec_loss(y_true, y_pred):

        lambda_obj = {{uniform(0,10)}}
        lambda_coord = {{uniform(0,10)}}
        lambda_iou = {{uniform(0,10)}}

        obj_loss = compute_obj_loss(y_true, y_pred)
        obj_loss = tf.multiply(lambda_obj,obj_loss)

        coord_loss = compute_coord_loss(y_true, y_pred)
        coord_loss = tf.multiply(lambda_coord,coord_loss)

        iou_loss = compute_iou_loss(y_true, y_pred)
        iou_loss = tf.multiply(lambda_iou,iou_loss)

        total_loss = tf.add(tf.add(obj_loss,coord_loss),iou_loss)

        return tf.reduce_mean(total_loss)


    def conv_batch_lrelu(input_tensor, numfilter, dim, strides=2):
        # https://github.com/guigzzz/Keras-Yolo-v2/blob/f61286371cdc2d470e0811234f552c70bbd5caba/yolo_layer_utils.py#L18
        input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                            kernel_regularizer=regularizers.l2(0.0005),
                            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                            use_bias=False
                        )(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)
        return LeakyReLU(alpha=0.1)(input_tensor)

    def convt_batch_lrelu(input_tensor, numfilter, dim, strides=2):
        input_tensor = Conv2DTranspose(numfilter, (dim, dim), strides=strides, padding='same',
                            kernel_regularizer=regularizers.l2(0.0005),
                            kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                            use_bias=False
                        )(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)
        return LeakyReLU(alpha=0.1)(input_tensor)

    MASK = 'MASK'
    VECTOR = 'VECTOR'

    inputs = Input(shape=(szx,szy,1))
    
    dropoutrate = {{uniform(0, 0.5)}}
    # down sample
    xd = conv_batch_lrelu(inputs, {{choice([16,32,64,128])}}, 3)
    xd=Dropout(dropoutrate)(xd)
    xd = conv_batch_lrelu(xd, {{choice([16,32,64,128])}}, 3)
    xd=Dropout(dropoutrate)(xd)
    xd = conv_batch_lrelu(xd, {{choice([16,32,64,128])}}, 3)
    xd=Dropout(dropoutrate)(xd)
    xd = conv_batch_lrelu(xd, {{choice([16,32,64,128])}}, 3)
    xd=Dropout(dropoutrate)(xd)

    # bottle neck
    x=Flatten()(xd)
    x=Dense({{choice([16,32,64,128,256])}},
        kernel_regularizer=regularizers.l2(0.0005),
        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
        use_bias=False,
        )(x) # # filter size ratio between conv and dense needs to be tuned!
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Dropout(dropoutrate)(x)
    x=Dense({{choice([16,32,64,128,256])}},
        kernel_regularizer=regularizers.l2(0.0005),
        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
        use_bias=False,
        )(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Dropout(dropoutrate)(x) # attempt to prevent overfit...
    x=Dense(smx*smy*5,
        kernel_regularizer=regularizers.l2(0.0005),
        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
        use_bias=True,
        )(x)
    x=Reshape((smx*smy,5))(x)

    # output split - split output to have varying activations for pos,width, and obj
    # output: pos and width
    x_pos_width = Lambda(lambda x: x[..., 0:4])(x)
    x_pos_width=Activation('sigmoid')(x_pos_width)
    #x_pos_width=Activation('linear')(x_pos_width)
    #x_pos_width=Lambda(lambda x: K.clip(x,0.,1.))(x_pos_width)

    # output: sigmoid
    x_obj = Lambda(lambda x: K.expand_dims(x[..., -1],axis=-1))(x)
    x_obj=Activation('sigmoid')(x_obj)

    # merge outputs
    vecs = Concatenate(axis=-1,name=VECTOR)([x_pos_width,x_obj])
    model = Model(inputs=inputs, outputs=vecs)
    model.summary()


    # very nice ref. https://github.com/ksanjeevan/dourflow/blob/master/yolov2.py
    # opt.Adam, opt.RMSprop
    batch_size={{choice([8,16,32,64,128,256])}}
    epochs=50
    lr={{uniform(1E-5,1E-3)}}
    decay={{uniform(1E-8,1E-2)}}
    patience=10
    beta_1=0.9
    beta_2=0.999
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  mode='min')
    callbacks = [early_stopping]

    optimizer = opt.Adam(lr=lr,beta_1=beta_1, beta_2=beta_2, decay=decay)
    model.compile(loss=vec_loss, optimizer=optimizer)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test),verbose=0)

    loss = model.evaluate(x_test, y_test,verbose=0)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

def data():
    x_train, y_train, _ = make_data(N=1024)
    x_test, y_test, _ = make_data(N=128)
    return x_train, y_train, x_test, y_test

best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=50,
                                      trials=Trials(),
                                      notebook_name='proto-line-detect-yolo')
x_train,y_train,x_val,y_val=data()

best_model.save('line-yolo-model.h5')
print(best_run)
print("Evalutation of best performing model:")
print(best_model.evaluate(x_val, y_val))
