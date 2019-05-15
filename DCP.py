# -*- coding: utf-8 -*-

import cv2
import numpy as np
import guidedfilter
import os


def get_dark_channel(I, w):
    
    M, N, _ = I.shape
    padded = np.pad(I, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    
    return darkch

def get_atmosphere(I, darkch, p):
    
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel() #arranged horizontally
    searchidx = (-flatdark).argsort()[:round(M * N * p)]
 
    return np.max(flatI.take(searchidx, axis=0), axis=0)

def get_transmission(I, A, darkch, omega, w):
    
    return 1 - omega * get_dark_channel(I / A, w)  # CVPR09, eq.12

def get_radiance(I, A, t):
    
    tiledt = np.zeros_like(I)  # tiled to M * N * 3
    tiledt[:, :, 0] = tiledt[:, :, 1] = tiledt[:, :, 2] = t
    return (I - A) / tiledt + A  # CVPR09, eq.16

def dehaze(im, tmin, w, p,
           omega, r, eps, L):
    
    I = np.asarray(im, dtype=np.float64)
    
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)
    A = get_atmosphere(I, Idark, p)
#    A = np.minimum(A, Amax)
    rawt = get_transmission(I, A, Idark, omega, w)
    rawt = np.maximum(rawt, tmin)
    normI = (I - I.min()) / (I.max() - I.min())  # normalize I
    refinedt = guidedfilter.guided_filter(normI, rawt, r, eps)

    clear_image = get_radiance(I, A, refinedt)
    
    return im, np.maximum(np.minimum(clear_image, L - 1), 0).astype(np.uint8) 

def use_dcp(hazy_image, p = 0.001, W = 16, omega = 0.95, L = 256):
    '''
    p      percent of pixels
    W      window size
    omega  before transmission
    L      highest pixel value
    '''
    _, im_dehaze = dehaze(hazy_image, tmin = 0.1, w=W, p=p,
       omega=omega, r=40, eps=1e-3, L = L)
    
    return im_dehaze

if __name__ =="__main__":
    
    images_path = '/home/jianan/Incoming/dongqin/test_real_images'
    #hazy_images, clear_images = use_dcp(images_path)







































