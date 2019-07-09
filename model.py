import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import regularizers

smooth = 1

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
#     print('dice_coef_loss')
    return 1-dice_coef(y_true, y_pred)

def unet_v1(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.5)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.5)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.5)(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_v1_deeper(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.5)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.5)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.5)(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    
    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(drop6))
    
    merge7 = concatenate([drop5,up7], axis = 3)
    conv7 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([drop4,up8], axis = 3)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([drop3,up9], axis = 3)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)

    up10 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv9))
    
    merge10 = concatenate([drop2,up10], axis = 3)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv10)
    conv10 = BatchNormalization()(conv10)
    
    up11 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv10))
    
    merge11 = concatenate([drop1,up11], axis = 3)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv11)
    conv11 = BatchNormalization()(conv11)
    
    conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv11)
    conv11 = BatchNormalization()(conv11)
    
    conv12 = Conv2D(1, 1, activation = 'sigmoid')(conv11)

    model = Model(input = inputs, output = conv12)
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 3e-5), loss = dice_coef_loss, metrics = [dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_v1_multi(pretrained_weights = None, input_size = (256,256,3), gpu_num = 2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(inputs)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv1)
    drop1 = Dropout(0.2)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv2)
    drop2 = Dropout(0.3)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.4)(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv4)
    
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(pool4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv5)
    
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge6)
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv6)
    

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge7)
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv7)
    

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge8)
    
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv8)
    

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(merge9)
    
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv9)
    
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.01))(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    parallel_model = multi_gpu_model(model, gpu_num)
    parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	parallel_model.load_weights(pretrained_weights)

    return parallel_model

