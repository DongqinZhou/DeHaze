# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import os
#import keras
from keras.layers import Conv2D, Concatenate, Multiply, Subtract, Add
from keras.models import Sequential
import keras.backend as K
from keras import optimizers





################### modified from data_input.py, test there!
def load_data(path,norm_height, norm_width):
    """
    load data and divide them into data-label pair.
    first read the file, then resize them, and eventually normalize them into [0,1] interval
    ideally, the input parameter should have a label path... but how could we keep the data and labels in 
    accordance? shuffle all files from one folder and then find its corresponding label in another folder?
    
    implementation is like this: 
    im_path = "H:\\Undergraduate\\18-19-3\\Undergraduate Thesis\\Dataset\\test_images"
    norm_height = 400
    norm_width = 100
    data, label = load_data(im_path, norm_height, norm_width)
    """
    data = []
    label = []
    files = os.listdir(path)
    random.seed(0)  #保证每次数据顺序一致
    random.shuffle(files)  #将所有的文件路径打乱
    for file in files:
        image = cv2.imread(path + "\\" + file)#读取文件
        image = cv2.resize(image,(norm_height,norm_width))#统一图片尺寸
        #cv2.imshow('image',image)
        #cv2.waitKey(0) 
        data.append(image)
        label.append(image)
    data = np.array(data,dtype="float")/255.0#归一化
    return data,label
    
################## define AOD-Net nodel
    
def AOD_Net_Model(channel,height,width):
    input_shape = (channel,height,width)
    if K.image_data_format() == "channels_last":#确认输入维度
        input_shape = (height,width,channel)
    model = Sequential()
    conv1 = Conv2D(3, 1, strides=(1, 1), padding='valid', activation='relu', input_shape = input_shape)
    model.add(conv1)
    conv2 = Conv2D(3, 3, strides=(1, 1), padding=(1,1), activation='relu')
    model.add(conv2)
    model.add(Concatenate([conv1, conv2], axis = -1))
    conv3 = Conv2D(3, 5, strides=(1, 1), padding=(2,2), activation='relu')
    model.add(conv3)
    model.add(Concatenate([conv2, conv3], axis = -1))
    conv4 = Conv2D(3, 7, strides=(1, 1), padding=(3,3), activation='relu')
    model.add(conv4)
    model.add(Concatenate([conv1, conv2, conv3, conv4], axis = -1))
    conv5 = Conv2D(3, 3, strides=(1, 1), padding=(1,1), activation='relu')
    model.add(conv5)   # here we've had the K output, how should we proceed?
    prod = Multiply([conv5, conv5])
    model.add(prod)
    subtract = Subtract([prod, conv5])
    model.add(subtract)
    one = K.backend(1, "float", (height. width))
    model.add(Add([subtract, one]))
    return model
    
def my_loss(y_true, y_pred):
    err = np.sum((y_true.astype("float") - y_pred.astype("float")) ** 2)
    err /= float(y_true.shape[0] * y_true.shape[1])
    return err

sgd = optimizers.SGD(lr=0.01, clipvalue=0.1, momentum=0.9, decay=0.001, nesterov=False)
model.compile(optimizers = sgd, loss = my_loss)
model.fit(x_train, y_train, epochs = 20, batch_size = 30, validation_data = (x_val, y_val))

                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     )





























