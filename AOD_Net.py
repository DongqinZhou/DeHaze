import cv2
import numpy as np
import random
import os

from keras.layers import Conv2D, Input, concatenate, multiply, subtract, Lambda
from keras import optimizers
from keras.models import Model
from keras.activations import relu 

 
def load_data(data_files,label_files, height, width):
   
    data = []
    label = []
    
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "/" + data_file)
        label_file = label_files[label_files.index(data_file[0:4] + data_file[-4:])]
        clear_image = cv2.imread(label_path + "/" + label_file)
        
        if hazy_image.shape != (height, width, 3):
            hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
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

def aodmodel():
    input_image = Input(shape = (None, None, 3), name = 'input')
    conv1 = Conv2D(3, (1,1), strides=(1, 1), padding='valid', activation='relu',kernel_initializer='random_normal', name = 'conv1')(input_image)
    conv2 = Conv2D(3, (3,3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'conv2')(conv1)
    concat1 = concatenate([conv1, conv2], axis = -1, name = 'concat1')
    conv3 = Conv2D(3, (5,5), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'conv3')(concat1)
    concat2 = concatenate([conv2, conv3], axis = -1, name = 'concat2')
    conv4 = Conv2D(3, (7,7), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'conv4')(concat2)
    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=-1, name = 'concat3')
    conv5 = Conv2D(3, (3,3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal', name = 'conv5')(concat3)
    prod = multiply([conv5, input_image], name = 'prod')
    diff = subtract([prod, conv5], name = 'diff')
    add_b = Lambda(lambda x: 1+x)(diff)
    out_image=Lambda(lambda x:relu(x))(add_b)
    model = Model(inputs = input_image, outputs = out_image)
    return model

def train_model(data_path, label_path, weights_path, lr=0.001, batch_size=32, p_train=0.8, width=320, height=240, nb_epochs=15):
    model = aodmodel()
    model.summary()
    sgd = optimizers.SGD(lr, clipvalue=0.1, momentum=0.9, decay=0.0001, nesterov=False)
    model.compile(optimizer = sgd, loss = 'mean_squared_error')
    
    p_train = p_train # proportion of training data
    width = width
    height = height
    batch_size = batch_size
    nb_epochs = nb_epochs
    
    data_files = os.listdir(data_path) 
    label_files = os.listdir(label_path)
    random.seed(100)  # ensure we have the same shuffled data every time
    random.shuffle(data_files)  
    data_files = data_files[0:40000]
    x_train = data_files[0: round(len(data_files) * p_train)]
    x_val =  data_files[round(len(data_files) * p_train) : len(data_files)]
    if len(x_train) % batch_size == 0:
        steps_per_epoch = len(x_train) // batch_size
    else:
        steps_per_epoch = len(x_train) // batch_size + 1
        
    if len(x_val) % batch_size == 0:
        steps = len(x_val) // batch_size
    else:
        steps = len(x_val) // batch_size + 1
    
    model.fit_generator(generator = get_batch(x_train, label_files, batch_size, height, width), 
                        steps_per_epoch=steps_per_epoch, epochs = nb_epochs, validation_data = 
                        get_batch(x_val, label_files, batch_size, height, width), validation_steps = steps,
                        use_multiprocessing=True, 
                        shuffle=False, initial_epoch=0)
    model.save_weights('aodnet.h5')
    print('model generated')
    return weights_path + '/aodnet.h5'

def Load_model(weights):
    model = aodmodel()
    model.load_weights(weights)
    return model

def usemodel(model, hazy_image):
    
    height = hazy_image.shape[0]
    width = hazy_image.shape[1]
    channel = hazy_image.shape[2]
    hazy_input = np.reshape(hazy_image, (1, height, width, channel)) / 255.0
    clear_ = model.predict(hazy_input)
    clear_image = np.floor(np.reshape(clear_, (height, width, channel)) * 255.0).astype(np.uint8)
    
    return clear_image

if __name__ =="__main__":
    
    ''' 
    Implementation of AOD-Net using keras.
    
    Usage: 
        1. modify the paths
        2. run this file
        3. visualize the images using:
            cv2.imshow('nameofwindow', hazy_image)
            cv2.imshow('nameofwindow', clear_image)
            cv2.waitKey(0)
            
    Parameter tuning:
        Changeable values are given in default, namely, lr, batch_size, p_train, nb_epochs
    '''
    im_path = ''
    data_path = ''
    label_path = ''
    weights_path = ''
    
    aod_weights = train_model(data_path, label_path, weights_path)

    aodnet = Load_model(aod_weights)    
    im = cv2.imread(im_path)
    im_dehaze = usemodel(aodnet, im)
