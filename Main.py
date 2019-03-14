# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:04:38 2019

@author: Zero_Zhou
"""

import cv2
import DCP_GF
im_path = "H:\\Undergraduate\\18-19-3\\Undergraduate Thesis\\Literature\\images\\Org images\\gugong.bmp"
im = cv2.imread(im_path)
cv2.imshow('image',im)
im_dehaze = DCP_GF.dehaze(im_path)
for i in range(5):
    cv2.imshow('image' + str(i), im_dehaze[i])
    
    
    
    

