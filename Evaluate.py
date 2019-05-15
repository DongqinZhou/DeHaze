# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np

from MSCNN import usemodel as MSCNN
from DehazeNet import usemodel as DehazeNet
from AOD_Net import usemodel as AOD_Net
from DCP import use_dcp as DCP
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


def PSNR(im_true, im_test):
    if im_true.shape != im_test.shape:
        im_true = cv2.resize(im_true, (im_test.shape[1], im_test.shape[0]), interpolation = cv2.INTER_AREA)
    return psnr(im_true, im_test)

def SSIM(im1, im2):
    if im1.shape != im2.shape:
        im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation = cv2.INTER_AREA)
    return ssim(im1, im2, multichannel = True, gaussian_weights = True)

### Video processing
def extract_video_frames(video_path, video_frames_path):
    '''
    Read a video file, and store them into a folder, while returning all frames of this video
    video_path: file path, something like .../XXX.mp4
    video_frames_path: folder path, something like .../video_1_frames
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 1
    success = True
    
    while(success):
        success, frame = cap.read()
        if success == False:
            break
        frames.append(frame)
        cv2.imwrite(video_frames_path + '/frame' + '_%d.jpg' % frame_count, frame)
        frame_count += 1
        
    cap.release()
    
    return frames

def takenum(elem):
    return int(elem.partition('_')[2].partition('.')[0])

def frame_to_video(video_path, frame_path, fps = 30, shape = (1280, 720)):
    '''
    Produce a video from some images
    video_path: file path, something like .../XXX.avi
    frame_path: folder path, something lke .../video_3_frames
    '''
    fps = fps
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, shape)
    image_files = os.listdir(frame_path)
    image_files.sort(key = takenum)
    
    for image_file in image_files:
        image = cv2.imread(frame_path + "/" + image_file)
        video_writer.write(image)
    
    video_writer.release()

def compute_psnr_ssim():
    '''
    computes PSNR and SSIM for DCP, AOD, DehazeNet, MSCNN
    stores the dehazed images to a file for calculation of SSEQ and BLIINDS-II
    '''
    testdata_path = '/home/jianan/Incoming/dongqin/testdata'
    testlabel_path = '/home/jianan/Incoming/dongqin/testlabel'
    
    AOD_Net_Weights = '/home/jianan/Incoming/dongqin/DeHaze/aodnet.h5'
    MSCNN_Coarse_Weights = '/home/jianan/Incoming/dongqin/DeHaze/coarseNet.h5'
    MSCNN_Fine_Weights = '/home/jianan/Incoming/dongqin/DeHaze/fineNet.h5'
    DehazeNet_Weights = '/home/jianan/Incoming/dongqin/DeHaze/dehazenet.h5'
    
    Hazy_Images_Path = '/home/jianan/Incoming/dongqin/Hazy_Images'
    Clear_Images_Path = '/home/jianan/Incoming/dongqin/Clear_Images'
    DCP_Dehazed_Path = '/home/jianan/Incoming/dongqin/DCP_Dehazed'
    AOD_Dehazed_Path = '/home/jianan/Incoming/dongqin/AOD_Dehazed'
    MSCNN_Dehazed_Path = '/home/jianan/Incoming/dongqin/MSCNN_Dehazed'
    DehazeNet_Dehazed_Path = '/home/jianan/Incoming/dongqin/DehazeNet_Dehazed'
    
    test_data_files = os.listdir(testdata_path)
    test_label_files = os.listdir(testlabel_path)
    random.shuffle(test_data_files)
    
    DCP_PSNR = []
    DCP_SSIM = []
    AOD_PSNR = []
    AOD_SSIM = []
    MSCNN_PSNR = []
    MSCNN_SSIM = []
    DehazeNet_PSNR = []
    DehazeNet_SSIM = []
    
    image_count = 1
    
    for test_data in test_data_files:
        test_label = test_label_files[test_label_files.index(test_data[0:4] + test_data[-4:])] # this is subject to change depending on the test set used
        hazy_image = cv2.imread(testdata_path + '/' + test_data)
        clear_image = cv2.imread(testlabel_path + '/' + test_label)
        
        DCP_Dehazed = DCP(hazy_image)
        AOD_Dehazed = AOD_Net(AOD_Net_Weights, hazy_image)
        DehazeNet_Dehazed = DehazeNet(DehazeNet_Weights, hazy_image)
        MSCNN_Dehazed = MSCNN(MSCNN_Coarse_Weights, MSCNN_Fine_Weights, hazy_image)
        
        DCP_PSNR.append(PSNR(clear_image, DCP_Dehazed))
        DCP_SSIM.append(SSIM(clear_image, DCP_Dehazed))
        AOD_PSNR.append(PSNR(clear_image, AOD_Dehazed))
        AOD_SSIM.append(SSIM(clear_image, AOD_Dehazed))
        MSCNN_PSNR.append(PSNR(clear_image, MSCNN_Dehazed))
        MSCNN_SSIM.append(SSIM(clear_image, MSCNN_Dehazed))
        DehazeNet_PSNR.append(PSNR(clear_image, DehazeNet_Dehazed))
        DehazeNet_SSIM.append(SSIM(clear_image, DehazeNet_Dehazed))
        
        cv2.imwrite(Hazy_Images_Path + '/Hazy_%d.jpg' % image_count, hazy_image)
        cv2.imwrite(Clear_Images_Path + '/Clear_%d.jpg' % image_count, clear_image)
        cv2.imwrite(DCP_Dehazed_Path + '/DCP_%d.jpg' % image_count, DCP_Dehazed)
        cv2.imwrite(AOD_Dehazed_Path + '/AOD_%d.jpg' % image_count, AOD_Dehazed)
        cv2.imwrite(MSCNN_Dehazed_Path + '/MSCNN_%d.jpg' % image_count, MSCNN_Dehazed)
        cv2.imwrite(DehazeNet_Dehazed_Path + '/DehazeNet_%d.jpg' % image_count, DehazeNet_Dehazed)
        
        image_count += 1
        
    return np.mean(DCP_PSNR), np.mean(DCP_SSIM), np.mean(AOD_PSNR), np.mean(AOD_SSIM), np.mean(MSCNN_PSNR), np.mean(MSCNN_SSIM), np.mean(DehazeNet_PSNR), np.mean(DehazeNet_SSIM)      
            

if __name__ =="__main__":
    
    dcp_psnr, dcp_ssim, aod_psnr, aod_ssim, mscnn_psnr, mscnn_ssim, dehazenet_psnr, dehazenet_ssim = compute_psnr_ssim()










