import os
import cv2
import random
import numpy as np
from keras.models import load_model

if __name__ == "__main__":
    #batch_size = 32
    
    model = load_model('aodnet.model')
    testdata_path = '/home/jianan/Desktop/dongqin_temp/Dataset/OTS002'
    testdata_files = os.listdir(testdata_path)
    random.seed(0)
    random.shuffle(testdata_files)
    testdata_files = testdata_files[0:128]
    #steps = len(testdata_files) // batch_size
    
    results = []
    for testdata_file in testdata_files:
        hazy_image = cv2.imread(testdata_path + '/' + testdata_file) / 255
        height = hazy_image.shape[0]
        width = hazy_image.shape[1]
        channel = hazy_image.shape[2]
        hazy_input = np.reshape(hazy_image, (1, height, width, channel))
        clear_image = model.predict(hazy_input)
        clear_output = np.floor(np.reshape(clear_image, (height, width, channel)) * 255).astype(np.uint8)
        results.append(clear_output)
    

'''
data_path = '/home/jianan/Desktop/dongqin_temp/Dataset/OTS002'
data_files = os.listdir(data_path)[0:10]
data = AOD_Net_2.load_testdata(data_files, 413,550)
'''