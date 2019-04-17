import cv2
import numpy as np
import random
import os

from keras.layers import Conv2D, Input, UpSampling2D, concatenate, multiply, subtract, Lambda, MaxPooling2D
from keras import optimizers
from keras.models import Model
import keras.backend as K
from keras.activations import relu


def load_data(data_files,label_files, height, width):
    
    data = []
    label = []
    
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "/" + data_file)
        if hazy_image.shape != (height, width, 3):
            hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
        label_file = label_files[label_files.index(data_file[0:4] + data_file[-4:])]
        clear_image = cv2.imread(label_path + "/" + label_file)
        if clear_image.shape != (height, width, 3):
            clear_image = cv2.resize(clear_image, (width, height), interpolation = cv2.INTER_AREA)
        data.append(hazy_image)
        label.append(clear_image)
    
    data = np.asarray(data) / 255.0
    label = np.asarray(label) / 255.0
    
    return data, label

def get_batch(data_files, label_files, batch_size, height, width):
   
    while 1:
        for i in range(0, len(data_files), batch_size):
            x, y = load_data(data_files[i : i+batch_size], label_files, height, width)
            
            yield x, y

def coarse_net():
    input_image = Input(shape = (None, None, 3))
    conv1 = Conv2D(5, (11,11), strides=(1, 1), padding='valid', activation='relu',kernel_initializer='random_normal')(input_image)
    mp1 = MaxPooling2D(pool_size = (2,2), padding = 'valid')(conv1)
    up1 = UpSampling2D(size=(2,2), interpolation = 'nearest')(mp1)
    conv2 = Conv2D(5, (9,9), strides=(1, 1), padding='valid', activation='relu',kernel_initializer='random_normal')(up1)
    mp2 = MaxPooling2D(pool_size = (2,2), padding = 'valid')(conv2)
    up2 = UpSampling2D(size=(2,2), interpolation = 'nearest')(mp2)
    conv3 = Conv2D(10, (7,7), strides=(1, 1), padding='valid', activation='relu',kernel_initializer='random_normal')(up2)
    mp3 = MaxPooling2D(pool_size = (2,2), padding = 'valid')(conv3)
    up3 = UpSampling2D(size=(2,2), interpolation = 'nearest')(mp3)
    #linear combination has a sigmoid!
    























