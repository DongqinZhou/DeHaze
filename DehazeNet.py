# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import os
import math
import keras.backend as K

from keras.layers import Conv2D, Input, concatenate, MaxPooling2D, Activation
from keras import optimizers, initializers
from keras.models import Model
from guidedfilter import guided_filter
from keras.engine.topology import Layer
from keras.callbacks import LearningRateScheduler
from keras.utils.generic_utils import get_custom_objects

def load_data(data_files,label_files, patch_size = 16):
    
    data = []
    label = []
    
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "/" + data_file)
        height = hazy_image.shape[0]
        width = hazy_image.shape[1]
        if height % patch_size != 0:
            height = height // patch_size * patch_size
        if width % patch_size != 0:
            width = width // patch_size * patch_size
        
        hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
        label_file = label_files[label_files.index(data_file.partition('.')[0][:-2] + data_file[-4:])]  #This is subject to modification depending on the file names of data_files.
        trans_map = cv2.imread(label_path + "/" + label_file, 0)
        trans_map = cv2.resize(trans_map, (width, height), interpolation = cv2.INTER_AREA)
        for i in random.sample(range(height // patch_size), height // patch_size):
            for j in random.sample(range(width // patch_size),width // patch_size):
                hazy_patch = hazy_image[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16), :]
                trans_patch = trans_map[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16),]
                data.append(hazy_patch)
                label.append(np.mean(trans_patch))
    
    data = np.asarray(data)  / 255.0
    label = np.asarray(label).reshape(len(label), 1, 1, 1) / 255.0
    
    return data, label

def get_batch(data_files, label_files, batch_size):
   
    while 1:
        for i in range(0, len(data_files), batch_size):
            x, y = load_data(data_files[i : i+batch_size], label_files)
            
            yield x, y

def BReLu(x):
    '''
    a self-defined activation function
    '''
    return K.minimum(K.maximum(0., x), 1.)

def get_airlight(hazy_image, trans_map, p):
    M, N = trans_map.shape
    flat_image = hazy_image.reshape(M*N, 3)
    flat_trans = trans_map.ravel()
    searchidx = (-flat_trans).argsort()[:round(M * N * p)]
    
    return np.max(flat_image.take(searchidx, axis=0), axis = 0)

def get_radiance(hazy_image, airlight, trans_map, L):
    tiledt = np.ones_like(hazy_image) * 0.1
    tiledt[:,:,0] = tiledt[:,:,1] = tiledt[:,:,2] = trans_map
    min_t = np.ones_like(hazy_image) * 0.2
    t = np.maximum(tiledt, min_t)
    
    hazy_image = hazy_image.astype(int)
    airlight = airlight.astype(int)
    airlight = np.minimum(airlight, 220)
    
    clear_ = (hazy_image - airlight) / t + airlight
    clear_image = np.maximum(np.minimum(clear_, L-1), 0).astype(np.uint8)
    
    return clear_image

class MaxoutConv2D(Layer):
    """
    Convolution Layer followed by Maxout activation as described 
    in https://arxiv.org/abs/1505.03540.
    
    Parameters
    ----------
    
    kernel_size: kernel_size parameter for Conv2D
    output_dim: final number of filters after Maxout
    nb_features: number of filter maps to take the Maxout over; default=4
    padding: 'same' or 'valid'
    """
    
    def __init__(self, kernel_size, output_dim, nb_features=4, padding='valid', use_bias = True, **kwargs):
        
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.nb_features = nb_features
        self.padding = padding
        self.use_bias = use_bias
        super(MaxoutConv2D, self).__init__(**kwargs)

    def call(self, x):

        output = None
        for _ in range(self.output_dim):
            
            conv_out = Conv2D(self.nb_features, self.kernel_size, padding=self.padding, use_bias = self.use_bias, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001))(x)            
            maxout_out = K.max(conv_out, axis=-1, keepdims=True)

            if output is not None:
                output = K.concatenate([output, maxout_out], axis=-1)

            else:
                output = maxout_out
        
        return output

    def compute_output_shape(self, input_shape):
        input_height= input_shape[1]
        input_width = input_shape[2]
        
        if(self.padding == 'same'):
            output_height = input_height
            output_width = input_width
        
        elif(input_height == None or input_width == None):
            return (input_shape[0], None, None, self.output_dim)
        
        else:
            output_height = input_height - self.kernel_size[0] + 1
            output_width = input_width - self.kernel_size[1] + 1
        
        return (input_shape[0], output_height, output_width, self.output_dim)
    
def DehazeNet(): #### carefully inspect the weights! this and all other networks!
    get_custom_objects().update({'BReLU':Activation(BReLu)})
    
    input_image = Input(shape = (None, None, 3), name = 'input')
    convmax = MaxoutConv2D(kernel_size = (5, 5), output_dim = 4, nb_features = 16, padding = 'valid', use_bias = False, name='convmax')(input_image)
    conv1 = Conv2D(16, (3, 3), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv1')(convmax)
    conv2 = Conv2D(16, (5, 5), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv2')(convmax)
    conv3 = Conv2D(16, (7, 7), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv3')(convmax)
    concat = concatenate([conv1, conv2, conv3], axis=-1, name='concat')
    mp = MaxPooling2D(pool_size=(7,7), strides=1, padding='valid', name='maxpool')(concat)
    conv4 = Conv2D(1, (6,6), padding='valid',activation='BReLU', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv4')(mp)
    
    model = Model(inputs = input_image, outputs = conv4)
    
    return model

def train_model(data_path, label_path, weights_path, lr=0.005, momentum=0.9, decay=5e-4, p_train = 0.8, batch_size = 100, nb_epochs = 50):
    
    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(sgd.lr)
            K.set_value(sgd.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(sgd.lr)

    dehazenet = DehazeNet()
    dehazenet.summary()
    
    sgd = optimizers.SGD(lr, momentum, decay, nesterov=False)
    dehazenet.compile(optimizer = sgd, loss = 'mean_squared_error')
                        
    data_files = os.listdir(data_path) 
    label_files = os.listdir(label_path)    
    
    random.seed(100)  # ensure we have the same shuffled data every time
    random.shuffle(data_files) 
    x_train = data_files[0: round(len(data_files) * p_train)]
    x_val =  data_files[round(len(data_files) * p_train) : len(data_files)]
    
    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    steps = math.ceil(len(x_val) / batch_size)
        
    reduce_lr = LearningRateScheduler(scheduler)
   
    dehazenet.fit_generator(generator = get_batch(x_train, label_files, batch_size), 
                        steps_per_epoch=steps_per_epoch, epochs = nb_epochs, validation_data = 
                        get_batch(x_val, label_files, batch_size), validation_steps = steps,
                        use_multiprocessing=True, 
                        shuffle=False, initial_epoch=0, callbacks = [reduce_lr])
    dehazenet.save_weights(weights_path + '/dehazenet.h5')
    print('dehazenet generated')
    
    return weights_path + '/dehazenet_weights.h5'

def Load_model(weights):
    dehazenet = DehazeNet()
    dehazenet.load_weights(weights)
    return dehazenet
    
def usemodel(dehazenet, hazy_image):
   
    patch_size = 16
    p = 0.001
    L = 256
    
    height = hazy_image.shape[0]
    width = hazy_image.shape[1]
    channel = hazy_image.shape[2]
    
    if height % patch_size != 0:
        height = height // patch_size * patch_size
    if width % patch_size != 0:
        width = width // patch_size * patch_size
        
    hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
    trans_map = np.zeros((height, width))
    
    for i in range(height // patch_size):
        for j in range(width // patch_size):
            hazy_patch = hazy_image[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16), :]
            hazy_input = np.reshape(hazy_patch, (1, patch_size, patch_size, channel)) / 255.0
            trans = dehazenet.predict(hazy_input)
            trans_map[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16)] = trans
    
    norm_hazy_image = (hazy_image - hazy_image.min()) / (hazy_image.max() - hazy_image.min())
    refined_trans_map = guided_filter(norm_hazy_image, trans_map)
    
    Airlight = get_airlight(hazy_image, refined_trans_map, p)
    clear_image = get_radiance(hazy_image, Airlight, refined_trans_map, L)
    
    return clear_image

if __name__ =="__main__":
    '''
    Implementation of DehazeNet using keras. https://arxiv.org/pdf/1601.07661.pdf
    
    Usage:
        1. modify the paths
        2. potentially change a line in load_data (see above)
        3. run this file
        4. visualize the images using:
            cv2.imshow('nameofwindow', hazy_image)
            cv2.imshow('nameofwindow', clear_image)
            cv2.waitKey(0)
            
    Parameter tuning:
        Changeable values are given in default, namely, lr, batch_size, p_train, nb_epochs etc.
    '''
    data_path = '/home/jianan/Incoming/dongqin/ITS_eg/haze'
    label_path = '/home/jianan/Incoming/dongqin/ITS_eg/trans'                      
    weights_path = '/home/jianan/Incoming/dongqin/ITS_eg'
    dehazenet_weights = train_model(data_path, label_path, weights_path)
    '''
    im_path = ''
    dehazenet = Load_model(dehazenet_weights)
    
    im = cv2.imread(im_path)
    im_dehaze = usemodel(dehazenet, im)
    '''