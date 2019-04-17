import os
import cv2
import random
import numpy as np
from keras.models import load_model
from keras.activations import relu 

if __name__ == "__main__":
    #batch_size = 32
    
    model = load_model('/home/jianan/Incoming/dongqin/DeHaze/AOD-Net/aodnet.model', custom_objects={"relu":relu})
    testdata_path = '/home/jianan/Incoming/dongqin/RTTS/JPEGImages'
    testdata_files = os.listdir(testdata_path)
    #random.seed(0)
    random.shuffle(testdata_files)
    testdata_files = testdata_files[0:20]
    #steps = len(testdata_files) // batch_size
    hazy_images = []
    clear_images = []
    for testdata_file in testdata_files:
        hazy_image = cv2.imread(testdata_path + '/' + testdata_file) / 255
        hazy_images.append(hazy_image)
        height = hazy_image.shape[0]
        width = hazy_image.shape[1]
        channel = hazy_image.shape[2]
        hazy_input = np.reshape(hazy_image, (1, height, width, channel))
        clear_image = model.predict(hazy_input)
        clear_output = np.floor(np.reshape(clear_image, (height, width, channel)) * 255).astype(np.uint8)
        clear_images.append(clear_output)
    for i in range(3):
        cv2.imshow('im'+str(i),hazy_images[i])
        cv2.imshow('im_'+str(i),clear_images[i])
    
    cv2.waitKey(0)


