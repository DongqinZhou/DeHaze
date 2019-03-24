
import cv2
import numpy as np
import random
import os

from keras.layers import Conv2D, Input, ZeroPadding2D, concatenate, add, multiply, subtract, Lambda
from keras import optimizers
from keras.models import Model
import keras.backend as K
#from keras.utils.vis_utils import plot_model







################### modified from data_input.py, test there!
def load_data(data_path,label_path, p_train, height, width):
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
        hazy_image = cv2.imread(data_path + "/" + data_file)#读取文件
        if hazy_image.shape != (height, width, 3):
            hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
        label_file = label_files[label_files.index(data_file[0:4] + data_file[-4:])]
        clear_image = cv2.imread(label_path + "/" + label_file)
        if clear_image.shape != (height, width, 3):
            clear_image = cv2.resize(clear_image, (width, height), interpolation = cv2.INTER_AREA)
        data.append(hazy_image)
        label.append(clear_image)
#    data = np.array(data,dtype="float")/255.0#归一化
    n_datapoint = len(data)
    x_train = np.asarray(data[0: round(n_datapoint * p_train)])
    x_test = np.asarray(data[round(n_datapoint * p_train):n_datapoint])
    y_train = np.asarray(label[0: round(n_datapoint * p_train)])
    y_test = np.asarray(label[round(n_datapoint * p_train):n_datapoint]) 
    '''
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test = np.array(y_test)
    '''
    return x_train, y_train, x_test, y_test


################## define AOD-Net nodel using functional API
def aodmodel():
    input_image = Input(shape = (None, None, 3))
    conv1 = Conv2D(3, (1,1), strides=(1, 1), padding='valid', activation='relu')(input_image)
    zp1 = ZeroPadding2D(padding = (1,1))(conv1)
    conv2 = Conv2D(3, (3,3), strides=(1, 1), activation='relu')(zp1)
    concat1 = concatenate([conv1, conv2])
    zp2 = ZeroPadding2D(padding = (2,2))(concat1)
    conv3 = Conv2D(3, (5,5), strides=(1, 1), activation='relu')(zp2)
    concat2 = concatenate([conv2, conv3])
    zp3 = ZeroPadding2D(padding = (3,3))(concat2)
    conv4 = Conv2D(3, (7,7), strides=(1, 1), activation='relu')(zp3)
    concat3 = concatenate([conv1, conv2, conv3, conv4])
    zp4 = ZeroPadding2D(padding = (1,1))(concat3)
    conv5 = Conv2D(3, (3,3), strides=(1, 1), activation='relu')(zp4)
    prod = multiply([conv5, input_image])
    diff = subtract([prod, conv5])
    #out_image = diff + 1
    one = Lambda(lambda x: K.ones_like(x))(diff)
    out_image = add([diff, one])
    
    model = Model(inputs = input_image, outputs = out_image)
    #model.summary()
    return model

#plot_model(model, to_file='aodmodel.png')

'''
##################### loss function
def my_loss(y_true, y_pred):
    err = np.sum((y_true - y_pred) ** 2)
    err /= (y_true.shape[0] * y_true.shape[1])
    return err
'''

##################### optimizer
if __name__ =="__main__":
    
    model = aodmodel()
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.1, momentum=0.9, decay=0.001, nesterov=False)
    model.compile(optimizer = sgd, loss = 'mean_squared_error')
    p_train = 0.8
    width = 550
    height = 413
    
    
    data_path = '/home/jianan/Desktop/dongqin_temp/Dataset/OTS001'
    label_path = '/home/jianan/Desktop/dongqin_temp/Dataset/clear_images'                     
    x_train, y_train, x_test, y_test = load_data(data_path, label_path, p_train, height, width)
    model.fit(x_train, y_train, epochs = 20, batch_size = 32)
    MSE = model.evaluate(x_test, y_test,batch_size = 32)
    # use the trained model: model.predict(X_new)
    model.save('/home/jianan/Desktop/dongqin_temp/DeHaze/aodnet.model')













