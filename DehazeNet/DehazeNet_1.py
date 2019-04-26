# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import os
import keras.backend as K

from keras.layers import Conv2D, Input, concatenate, MaxPooling2D, Activation
from keras import optimizers, initializers
from keras.models import Model
from keras.engine.topology import Layer
from keras.callbacks import LearningRateScheduler
from keras.utils.generic_utils import get_custom_objects

def load_data(data_files,label_files, height, width, patch_size = 16):
    
    data = []
    label = []
    
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "/" + data_file)
        if hazy_image.shape != (height, width, 3):
            hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
        label_file = label_files[label_files.index(data_file.partition('.')[0][:-2] + data_file[-4:])]
        trans_map = cv2.imread(label_path + "/" + label_file, 0)
        if trans_map.shape != (height, width):
            trans_map = cv2.resize(trans_map, (width, height), interpolation = cv2.INTER_AREA)
        for i in random.sample(range(height // patch_size), height // patch_size):
            for j in random.sample(range(width // patch_size),width // patch_size):
                hazy_patch = hazy_image[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16), :]
                trans_patch = trans_map[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16),]
                data.append(hazy_patch)
                label.append(np.mean(trans_patch))
    
    data = np.asarray(data) # whether to normalize?
    label = np.asarray(label).reshape(len(label), 1, 1, 1) / 255.0
    
    return data, label

def get_batch(data_files, label_files, batch_size, height, width):
   
    while 1:
        for i in range(0, len(data_files), batch_size):
            x, y = load_data(data_files[i : i+batch_size], label_files, height, width)
            
            yield x, y
            
def scheduler(epoch):
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(sgd.lr)
        K.set_value(sgd.lr, lr - 0.1)
        print("lr changed to {}".format(lr - 0.1))
    return K.get_value(sgd.lr)

def BReLu(x):
    return K.minimum(K.maximum(0., x), 1.)

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
    first_layer: True if x is the input_tensor
    input_shape: Required if first_layer=True
    
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
            # make modifications here for weight initialization
            
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
    input_image = Input(shape = (None, None, 3), name = 'input')
    print(K.int_shape(input_image))
    convmax = MaxoutConv2D(kernel_size = (5, 5), output_dim = 4, nb_features = 16, padding = 'valid', use_bias = False, name='convmax')(input_image)
    print(K.int_shape(convmax))
    conv1 = Conv2D(16, (3, 3), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv1')(convmax)
    print(K.int_shape(conv1))
    conv2 = Conv2D(16, (5, 5), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv2')(convmax)
    print(K.int_shape(conv2))
    conv3 = Conv2D(16, (7, 7), padding = 'same', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv3')(convmax)
    print(K.int_shape(conv3))
    concat = concatenate([conv1, conv2, conv3], axis=-1, name='concat')
    print(K.int_shape(concat))
    mp = MaxPooling2D(pool_size=(7,7), strides=1, padding='valid', name='maxpool')(concat)
    print(K.int_shape(mp))
    conv4 = Conv2D(1, (6,6), padding='valid',activation='BReLU', use_bias = False, kernel_initializer=initializers.random_normal(mean=0.,stddev=0.001),name='conv4')(mp)
    print(K.int_shape(conv4))
    
    model = Model(inputs = input_image, outputs = conv4)
    
    return model

'''
data_path = '/home/jianan/Incoming/dongqin/ITS_eg/haze'
label_path = '/home/jianan/Incoming/dongqin/ITS_eg/trans'                      
data_files = os.listdir(data_path) # seems os reads files in an arbitrary order
label_files = os.listdir(label_path)   
data, label = load_data(data_files, label_files, 240, 320) 

'''
if __name__ =="__main__":
    
    sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=5e-4, nesterov=False)
    get_custom_objects().update({'BReLU':Activation(BReLu)})
    p_train = 0.8
    width = 320
    height = 240
    batch_size = 10
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    data_path = '/home/jianan/Incoming/dongqin/ITS_eg/haze'
    label_path = '/home/jianan/Incoming/dongqin/ITS_eg/trans'                      
    data_files = os.listdir(data_path) # seems os reads files in an arbitrary order
    label_files = os.listdir(label_path)    
    
    #random.seed(0)  # ensure we have the same shuffled data every time
    random.shuffle(data_files) 
    x_train = data_files[0: round(len(data_files) * p_train)]
    x_val =  data_files[round(len(data_files) * p_train) : len(data_files)]
    steps_per_epoch = len(x_train) // batch_size 
    steps = len(x_val) // batch_size 
    reduce_lr = LearningRateScheduler(scheduler)
    
    dehazenet = DehazeNet()
    dehazenet.summary()

    
    
    dehazenet.compile(optimizer = sgd, loss = 'mean_squared_error')
    dehazenet.fit_generator(generator = get_batch(x_train, label_files, batch_size, height, width), 
                        steps_per_epoch=steps_per_epoch, epochs = 2, validation_data = 
                        get_batch(x_val, label_files, batch_size, height, width), validation_steps = steps,
                        use_multiprocessing=True, 
                        shuffle=False, initial_epoch=0, callbacks = [reduce_lr])
    dehazenet.save_weights('dehazenet_weights.h5')
    print('dehazenet generated')

    
    
    













