# -*- coding: utf-8 -*-
import os
import cv2
import time
import random
import numpy as np

from MSCNN import usemodel as MSCNN
from MSCNN import Load_model as load_mscnn
from DehazeNet import usemodel as DehazeNet
from DehazeNet import Load_model as load_dehazenet
from AOD_Net import usemodel as AOD_Net
from AOD_Net import Load_model as load_aodnet
from DCP import dehaze_1 as DCP_1
from DCP import dehaze_2 as DCP_2
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

def extract_video_frames(video_path, video_frames_path):
    '''
    Break a video into discrete frames. Return the frames and store them into a folder.
    
    video_path:         file path
    video_frames_path:  folder path
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
    Generate a video from frames.
    
    video_path: file path
    frame_path: folder path
    '''
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, shape)
    image_files = os.listdir(frame_path)
    image_files.sort(key = takenum)
    
    for image_file in image_files:
        image = cv2.imread(frame_path + "/" + image_file)
        video_writer.write(image)
    
    video_writer.release()
    
def video_dehaze(fps, width, height):
    '''
    Read a video from video_path, store video frames in video_frames_path; dehaze frames and store dehazed frames in AOD_dehazed_frames_path; then generate a dehazed video and store it in dehazed_video_path.
    
    video path :                file path
    video_frames_path :         folder path
    AOD_dehazed_frames_path :   folder path
    dehazed_video_path :       file path
    '''
    video_path = ''
    video_frames_path = ''
    AOD_dehazed_frames_path = ''
    dehazed_video_path = ''
    AOD_Net_Weights = ''
    
    model_aod = load_aodnet(AOD_Net_Weights)
    
    hazy_images = extract_video_frames(video_path, video_frames_path)
    image_count = 1
    
    for hazy_image in hazy_images:
        AOD_Dehazed = AOD_Net(model_aod, hazy_image)
        cv2.imwrite(AOD_dehazed_frames_path + '/AOD_%d.jpg' % image_count, AOD_Dehazed)
        
        image_count += 1
    
    frame_to_video(dehazed_video_path + '/AOD_Dehazed_Video.avi', AOD_dehazed_frames_path, fps, shape = (width, height))
  
def compute_psnr_ssim():
    '''
    computes PSNR and SSIM for DCP, AOD, DehazeNet, MSCNN
    stores the dehazed images to a file for calculation of SSEQ and BRISQUE
    '''
    testdata_path = ''
    testlabel_path = ''
    
    AOD_Net_Weights = ''
    MSCNN_Weights = ''
    DehazeNet_Weights = ''
    
    model_aod = load_aodnet(AOD_Net_Weights)
    model_dehazenet = load_dehazenet(DehazeNet_Weights)
    model_mscnn = load_mscnn(MSCNN_Weights)
    
    Hazy_Images_Path = ''
    Clear_Images_Path = ''
    DCP_Dehazed_Path_1 = ''
    DCP_Dehazed_Path_2 = ''
    AOD_Dehazed_Path = ''
    MSCNN_Dehazed_Path = ''
    DehazeNet_Dehazed_Path = ''
    
    test_data_files = os.listdir(testdata_path)
    test_label_files = os.listdir(testlabel_path)
    random.shuffle(test_data_files)
    
    DCP_PSNR = []
    DCP_SSIM = []
    DCP_2_PSNR = []
    DCP_2_SSIM = []
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
        
        DCP_Dehazed_1 = DCP_1(hazy_image)
        DCP_Dehazed_2 = DCP_2(hazy_image)
        AOD_Dehazed = AOD_Net(model_aod, hazy_image)
        DehazeNet_Dehazed = DehazeNet(model_dehazenet, hazy_image)
        MSCNN_Dehazed = MSCNN(model_mscnn, hazy_image)
        
        DCP_PSNR.append(PSNR(clear_image, DCP_Dehazed_1))
        DCP_SSIM.append(SSIM(clear_image, DCP_Dehazed_1))
        DCP_2_PSNR.append(PSNR(clear_image, DCP_Dehazed_2))
        DCP_2_SSIM.append(SSIM(clear_image, DCP_Dehazed_2))
        AOD_PSNR.append(PSNR(clear_image, AOD_Dehazed))
        AOD_SSIM.append(SSIM(clear_image, AOD_Dehazed))
        MSCNN_PSNR.append(PSNR(clear_image, MSCNN_Dehazed))
        MSCNN_SSIM.append(SSIM(clear_image, MSCNN_Dehazed))
        DehazeNet_PSNR.append(PSNR(clear_image, DehazeNet_Dehazed))
        DehazeNet_SSIM.append(SSIM(clear_image, DehazeNet_Dehazed))
        
        cv2.imwrite(Hazy_Images_Path + '/Hazy_%d.jpg' % image_count, hazy_image)
        cv2.imwrite(Clear_Images_Path + '/Clear_%d.jpg' % image_count, clear_image)
        cv2.imwrite(DCP_Dehazed_Path_1 + '/DCP_%d.jpg' % image_count, DCP_Dehazed_1)
        cv2.imwrite(DCP_Dehazed_Path_2 + '/DCP_%d.jpg' % image_count, DCP_Dehazed_2)
        cv2.imwrite(AOD_Dehazed_Path + '/AOD_%d.jpg' % image_count, AOD_Dehazed)
        cv2.imwrite(MSCNN_Dehazed_Path + '/MSCNN_%d.jpg' % image_count, MSCNN_Dehazed)
        cv2.imwrite(DehazeNet_Dehazed_Path + '/DehazeNet_%d.jpg' % image_count, DehazeNet_Dehazed)
        
        image_count += 1
        
    return np.mean(DCP_PSNR), np.mean(DCP_SSIM), np.mean(DCP_2_PSNR), np.mean(DCP_2_SSIM), np.mean(AOD_PSNR), np.mean(AOD_SSIM), np.mean(MSCNN_PSNR), np.mean(MSCNN_SSIM), np.mean(DehazeNet_PSNR), np.mean(DehazeNet_SSIM) 

def run_time():
    '''
    Compare run time of each algorithm
    '''
    data_path = ''
    data_files = os.listdir(data_path)
    
    AOD_Net_Weights = ''
    MSCNN_Weights = ''
    DehazeNet_Weights = ''
    
    model_aod = load_aodnet(AOD_Net_Weights)
    model_dehazenet = load_dehazenet(DehazeNet_Weights)
    model_mscnn = load_mscnn(MSCNN_Weights)
    
    images = []
    
    for data_file in data_files:
        im = cv2.imread(data_path + '/' + data_file)
        images.append(im)
    
    dcp = time.clock()
    for i in range(len(images)):
        _ = DCP_2(images[i])
    dcp_average = (time.clock() - dcp) / len(images)
    
    print('DCP average time per image: ', dcp_average)
    
    dehazenet = time.clock()
    for i in range(len(images)):
        _ = DehazeNet(model_dehazenet, images[i])
    dehazenet_average = (time.clock() - dehazenet) / len(images)
    
    print('DehazeNet average time per image: ', dehazenet_average)
    
    mscnn = time.clock()
    for i in range(len(images)):
        _ = MSCNN(model_mscnn, images[i])
    mscnn_average = (time.clock() - mscnn) / len(images)
    
    print('MSCNN average time per image: ', mscnn_average)
    
    aod = time.clock()
    for i in range(len(images)):
        _ = AOD_Net(model_aod, images[i])
    aod_average = (time.clock() - aod) / len(images)
    
    print('AOD average time per image: ', aod_average)
    

if __name__ =="__main__":
    
    dcp_psnr, dcp_ssim, dcp_2_psnr, dcp_2_ssim, aod_psnr, aod_ssim, mscnn_psnr, mscnn_ssim, dehazenet_psnr, dehazenet_ssim = compute_psnr_ssim()
    #video_dehaze(30,1280,720)








