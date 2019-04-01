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
    tiledt[:, :, R] = tiledt[:, :, G] = tiledt[:, :, B] = t
    return (I - A) / tiledt + A  # CVPR09, eq.16

def dehaze(im_path, tmin, Amax, w, p,
           omega, r, eps):
    
    im = cv2.imread(im_path)
    I = np.asarray(im, dtype=np.float64)
    
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)
    A = get_atmosphere(I, Idark, p)
    A = np.minimum(A, Amax)
    rawt = get_transmission(I, A, Idark, omega, w)
    normI = (I - I.min()) / (I.max() - I.min())  # normalize I
    refinedt = guidedfilter.guided_filter(normI, rawt, r, eps)

    clear_image = get_radiance(I, A, refinedt)
    
    return np.maximum(np.minimum(clear_image, L - 1), 0).astype(np.uint8) 

if __name__ =="__main__":

    p = 0.001  # percent of pixels
    W = 16     # window size
    omega = 0.95 # omega before transmission
    R, G, B = 0, 1, 2  # index for convenience
    L = 256  # color depth
    images_path = r'H:\Undergraduate\18-19-3\Undergraduate Thesis\Dataset\test_images_data'
    images_filenames = os.listdir(images_path)
    for image_filename in images_filenames:
        image_path = images_path + '\\' + image_filename
        im_dehaze = dehaze(image_path, tmin=0.2, Amax=220, w=W, p=p,
           omega=omega, r=40, eps=1e-3)











































