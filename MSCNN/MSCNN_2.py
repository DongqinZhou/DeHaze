# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
from MSCNN_1 import MSCNN

def get_airlight(hazy_image, trans_map, p):
    M, N = trans_map.shape
    flat_image = hazy_image.reshape(M*N, 3)
    flat_trans = trans_map.ravel()
    searchidx = (-flat_trans).argsort()[:round(M * N * p)]
    
    return np.max(flat_image.take(searchidx, axis=0), axis = 0)

def get_radiance(hazy_image, airlight, trans_map, L):
    tiledt = np.zeros_like(hazy_image)
    tiledt[:,:,0] = tiledt[:,:,1] = tiledt[:,:,2] = trans_map
    min_t = np.ones_like(hazy_image) * 0.1
    t = np.maximum(tiledt, min_t)
    hazy_image = hazy_image.astype(int)
    airlight = airlight.astype(int)
    clear_ = (hazy_image - airlight) / t + airlight
    clear_image = np.maximum(np.minimum(clear_, L-1), 0).astype(np.uint8)
    
    return clear_image

if __name__ =="__main__":
    
    mscnn = MSCNN()
    mscnn.load_weights('mscnn.h5')
    
    testdata_path = '/home/jianan/Incoming/dongqin/OTS/OTS001'
    testdata_files = os.listdir(testdata_path)
    #random.seed(0)
    random.shuffle(testdata_files)
    testdata_files = testdata_files[0:20]
   
    height = 240
    width = 320
    p = 0.001
    L = 256
    
    hazy_images = []
    trans_maps = []
    clear_images = []
    
    for testdata_file in testdata_files:
        hazy_image = cv2.imread(testdata_path + '/' + testdata_file) 
        if hazy_image.shape != (height, width, 3):
                hazy_image = cv2.resize(hazy_image, (width, height), interpolation = cv2.INTER_AREA)
        hazy_images.append(hazy_image)
        height = hazy_image.shape[0]
        width = hazy_image.shape[1]
        channel = hazy_image.shape[2]
        hazy_input = np.reshape(hazy_image, (1, height, width, channel))
        trans_map = mscnn.predict(hazy_input)
        trans_map = np.floor(np.reshape(trans_map, (height, width)))
        trans_maps.append(trans_map)
        Airlight = get_airlight(hazy_image, trans_map, p)
        clear_image = get_radiance(hazy_image, Airlight, trans_map, L)
        clear_images.append(clear_image)

    
    for i in range(3):
        cv2.imshow('im'+str(i),hazy_images[i])
        cv2.imshow('im_'+str(i),trans_maps[i])
    
    cv2.waitKey(0)
    



























