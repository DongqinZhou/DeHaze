
import cv2
import numpy as np
import random
import os

from keras.layers import Conv2D, Input, concatenate, multiply, subtract, Lambda
from keras import optimizers
from keras.models import Model
import keras.backend as K
from keras.activations import relu 


################### 
def load_data(data_files,label_files, height, width):
    '''
    label_path: the path where all label images locate
    data_path: the path where a batch of hazy images locate 
    '''
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
#    data = np.array(data,dtype="float")/255.0#归一化
    
    data = np.asarray(data) / 255.0
    label = np.asarray(label) / 255.0
    
    return data, label

def get_batch(data_files, label_files, batch_size, height, width):
   
    while 1:
        for i in range(0, len(data_files), batch_size):
            x, y = load_data(data_files[i : i+batch_size], label_files, height, width)
            
            yield x, y

################## define AOD-Net nodel using functional API
def aodmodel():
    input_image = Input(shape = (None, None, 3))
    conv1 = Conv2D(3, (1,1), strides=(1, 1), padding='valid', activation='relu',kernel_initializer='random_normal')(input_image)
    #zp1 = ZeroPadding2D(padding = (1,1))(conv1)
    conv2 = Conv2D(3, (3,3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal')(conv1)
    concat1 = concatenate([conv1, conv2], axis = -1)
    #zp2 = ZeroPadding2D(padding = (2,2))(concat1)
    conv3 = Conv2D(3, (5,5), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal')(concat1)
    concat2 = concatenate([conv2, conv3], axis = -1)
    #zp3 = ZeroPadding2D(padding = (3,3))(concat2)
    conv4 = Conv2D(3, (7,7), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal')(concat2)
    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=-1)
    #zp4 = ZeroPadding2D(padding = (1,1))(concat3)
    conv5 = Conv2D(3, (3,3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='random_normal')(concat3)
    prod = multiply([conv5, input_image])
    diff = subtract([prod, conv5])
    #out_image = diff + 1
    add_b = Lambda(lambda x: 1+x)(diff)
    out_image=Lambda(lambda x:relu(x))(add_b)
    model = Model(inputs = input_image, outputs = out_image)
    #model.summary()
    return model

'''
##################### loss function
def my_loss(y_true, y_pred):
    err = np.sum((y_true - y_pred) ** 2)
    err /= (y_true.shape[0] * y_true.shape[1])
    return err
'''

if __name__ =="__main__":
    
    model = aodmodel()
    sgd = optimizers.SGD(lr=0.001, clipvalue=0.1, momentum=0.9, decay=0.0001, nesterov=False)
    model.compile(optimizer = sgd, loss = 'mean_squared_error')
    p_train = 0.7
    width = 320
    height = 240
    batch_size = 32
    
    data_path = '/home/jianan/Desktop/dongqin_temp/Dataset/OTS001'
    label_path = '/home/jianan/Desktop/dongqin_temp/Dataset/clear_images'                      
    data_files = os.listdir(data_path) # seems os reads files in an arbitrary order
    label_files = os.listdir(label_path)
    
    random.seed(0)  # ensure we have the same shuffled data every time
    random.shuffle(data_files)  
    x_train = data_files[0: round(len(data_files) * p_train)]
    x_val =  data_files[round(len(data_files) * p_train) : len(data_files)]
    steps_per_epoch = len(x_train) // batch_size + 1
    steps = len(x_val) // batch_size + 1
    
    model.fit_generator(generator = get_batch(x_train, label_files, batch_size, height, width), 
                        steps_per_epoch=steps_per_epoch, epochs = 20, validation_data = 
                        get_batch(x_val, label_files, batch_size, height, width), validation_steps = steps,
                        use_multiprocessing=True, 
                        shuffle=False, initial_epoch=0)
    # use the trained model: model.predict(X_new)
    model.save('/home/jianan/Desktop/dongqin_temp/DeHaze/aodnet.model')
    print('model generated')


'''
width = 550
height = 413
batch_size = 32

data_path = '/home/jianan/Desktop/dongqin_temp/Dataset/OTS001'
label_path = '/home/jianan/Desktop/dongqin_temp/Dataset/clear_images'                      
data_files = os.listdir(data_path) # seems os reads files in an arbitrary order
label_files = os.listdir(label_path)

random.seed(0)  # ensure we have the same shuffled data every time
random.shuffle(data_files)  
x_train = data_files[0: 10]

x, y = load_data(x_train, label_files, height, width)
'''






























