#!/usr/bin/env python
# encoding: utf-8
#from keras.preprocessing.image import img_to_array#图片转为array
#from keras.utils import to_categorical#相当于one-hot
#from imutils import paths
import cv2
import numpy as np
import random
import os

def load_data(path,norm_height, norm_width):
    data = []#数据x
    label = []#标签y
    files = os.listdir(path)
    random.seed(0)#保证每次数据顺序一致
    random.shuffle(files)#将所有的文件路径打乱
    for file in files:
        image = cv2.imread(path + "\\" + file)#读取文件
        image = cv2.resize(image,(norm_height,norm_width))#统一图片尺寸
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        data.append(image)
#        maker = int(each_path.split(os.path.sep)[-2])#切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        label.append(image)
    data = np.array(data,dtype="float")/255.0#归一化
#    label = np.array(label)
#    label = to_categorical(label,num_classes=class_num)#one-hot
    return data,label

im_path = "H:\\Undergraduate\\18-19-3\\Undergraduate Thesis\\Dataset\\test_images"
norm_height = 400
norm_width = 100
data, label = load_data(im_path, norm_height, norm_width)