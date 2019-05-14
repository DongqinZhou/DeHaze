# -*- coding: utf-8 -*-
import os
import cv2

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def PSNR(im_true, im_test):
    return psnr(im_true, im_test)

def SSIM(im1, im2):
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















