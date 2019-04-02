# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import os
#import keras
from keras.layers import Conv2D, Concatenate, Multiply, Subtract, Add, ZeroPadding2D
from keras.models import Sequential
import keras.backend as K
from keras import optimizers

'''
This model was defined using Sequential model, not so very suitable for this problem.
'''



################### modified from data_input.py, test there!
def load_data(data_path,label_path, p_train):
    """
    load data and divide them into data-label pair.
    first read the file, then resize them, and eventually normalize them into [0,1] interval
    ideally, the input parameter should have a label path... but how could we keep the data and labels in 
    accordance? shuffle all files from one folder and then find its corresponding label in another folder?
    
    p_train: proportion of training data
    data_path: path where hazy images locate
    label_path: path where clear images locate
    """
    data = []
    label = []
    data_files = os.listdir(data_path)
    label_files = os.listdir(label_path)
    random.seed(0)  #保证每次数据顺序一致
    random.shuffle(data_files)  #将所有的文件路径打乱
    for data_file in data_files:
        hazy_image = cv2.imread(data_path + "\\" + data_file)#读取文件
        label_file = label_files[label_files.index(data_file[0:4] + data_file[-4:])]
        clear_image = cv2.imread(label_path + "\\" + label_file)
        data.append(hazy_image)
        label.append(clear_image)
#    data = np.array(data,dtype="float")/255.0#归一化
    n_datapoint = len(data)
    x_train = np.asarray(data[0: round(n_datapoint * p_train)])
    x_test = np.asarray(data[round(n_datapoint * p_train):n_datapoint])
    y_train = np.asarray(label[0: round(n_datapoint * p_train)])
    y_test = np.asarray(label[round(n_datapoint * p_train):n_datapoint])    
    
    return x_train, y_train, x_test, y_test


################## define AOD-Net nodel using Sequential model
input_shape = (None,None,3)
model = Sequential()
conv1 = Conv2D(3, 1, strides=(1, 1), padding='valid', activation='relu', input_shape = input_shape)
model.add(conv1)
zp1 = ZeroPadding2D(padding = (1,1))
model.add(zp1)
conv2 = Conv2D(3, 3, strides=(1, 1), activation='relu')
model.add(conv2)
concat1 = Concatenate([conv1, conv2])
model.add(concat1)
zp2 = ZeroPadding2D(padding = (2,2))
model.add(zp2)
conv3 = Conv2D(3, 5, strides=(1, 1), activation='relu')
model.add(conv3)
concat2 = Concatenate([conv2, conv3])
model.add(concat2)
zp3 = ZeroPadding2D(padding = (3,3))
model.add(zp3)
conv4 = Conv2D(3, 7, strides=(1, 1), activation='relu')
model.add(conv4)
concat3 = Concatenate([conv1, conv2, conv3, conv4])
model.add(concat3)
zp4 = ZeroPadding2D(padding = (1,1))
model.add(zp4)
conv5 = Conv2D(3, 3, strides=(1, 1), activation='relu')
model.add(conv5)   # here we've had the K output, how should we proceed?
prod = Multiply([conv5, conv5])
model.add(prod)
subtract = Subtract([prod, conv5])
model.add(subtract)
one = K.backend(1, dtype="float")
model.add(Add([subtract, one]))

##################### loss function
def my_loss(y_true, y_pred):
    err = np.sum((y_true.astype("float") - y_pred.astype("float")) ** 2)
    err /= float(y_true.shape[0] * y_true.shape[1])
    return err

##################### optimizer
sgd = optimizers.SGD(lr=0.01, clipvalue=0.1, momentum=0.9, decay=0.001, nesterov=False)
model.compile(optimizer = sgd, loss = my_loss)
p_train = 0.8

data_path = r'H:\Undergraduate\18-19-3\Undergraduate Thesis\Dataset\test_image_data'
label_path = r'H:\Undergraduate\18-19-3\Undergraduate Thesis\Dataset\test_images_label'                     
x_train, y_train, x_test, y_test = load_data(data_path, label_path, p_train)
model.fit(x_train, y_train, epochs = 20, batch_size = 32)
MSE = model.evaluate(x_test, y_test,batch_size = 32)
# use the trained model: model.predict(X_new)
model.save('C:\\Users\\Zero_Zhou\\DeHaze\\AOD-Net\\aodnet.model')

                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                    





























