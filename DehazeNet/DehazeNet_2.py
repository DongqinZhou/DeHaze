# -*- coding: utf-8 -*-

# suppose dehazenet has been successfully trained, this is what we do next.

import os
import cv2
import random
import numpy as np
from DehazeNet_1 import DehazeNet, BReLu
from keras.layers import Activation
from guidedfilter import guided_filter
from keras.utils.generic_utils import get_custom_objects

def get_airlight(hazy_image, trans_map, p):
    M, N = trans_map.shape
    flat_image = hazy_image.reshape(M*N, 3)
    flat_trans = trans_map.ravel()
    searchidx = (-flat_trans).argsort()[:round(M * N * p)]
    
    return np.max(flat_image.take(searchidx, axis=0), axis = 0)

def get_radiance(hazy_image, airlight, trans_map, L):
    tiledt = np.zeros_like(hazy_image)
    tiledt[:,:,0] = tiledt[:,:,1] = tiledt[:,:,2] = trans_map
    #min_t = np.ones_like(hazy_image) * 0.1
    #t = np.maximum(tiledt, min_t)
    hazy_image = hazy_image.astype(int)
    airlight = airlight.astype(int)
    clear_ = (hazy_image - airlight) / tiledt + airlight
    clear_image = np.maximum(np.minimum(clear_, L-1), 0).astype(np.uint8)
    
    return clear_image


if __name__ =="__main__":
    
    get_custom_objects().update({'BReLU':Activation(BReLu)})
    dehazenet = DehazeNet()
    dehazenet.load_weights('dehazenet_weights.h5')
    
    testdata_path = '/home/jianan/Incoming/dongqin/OTS/OTS001'
    testdata_files = os.listdir(testdata_path)
    #random.seed(0)
    random.shuffle(testdata_files)
    testdata_files = testdata_files[0:20]
   
    height = 240
    width = 320
    patch_size = 16
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
        trans_map = np.zeros((height, width))
        
        for i in range(height // patch_size):
            for j in range(width // patch_size):
                hazy_patch = hazy_image[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16), :]
                hazy_input = np.reshape(hazy_patch, (1, patch_size, patch_size, channel))
                trans = dehazenet.predict(hazy_input)
                trans_map[(i * 16) : (16 * i + 16), (j * 16) : (j * 16 + 16)] = trans
        
        norm_hazy_image = (hazy_image - hazy_image.min()) / (hazy_image.max() - hazy_image.min())
        refined_trans_map = guided_filter(norm_hazy_image, trans_map)
        
        trans_maps.append(refined_trans_map)
        
        Airlight = get_airlight(hazy_image, trans_map, p)
        clear_image = get_radiance(hazy_image, Airlight, trans_map, L)
        clear_images.append(clear_image)

    
    for i in range(3):
        cv2.imshow('im'+str(i),hazy_images[i])
        cv2.imshow('im_'+str(i),trans_maps[i])
    
    cv2.waitKey(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

