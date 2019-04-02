import cv2
import numpy as np
'''
def resize_image(image, width, height):
    resized_image = image
    if image.shape != (height, width, 3):
        resized_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resized_image
'''



def norm_image(image):  # This function is useless. image / 255.0 does the same job.
    normed_image = np.ones(shape = image.shape)
    for i in range(3):
        normed_image[:,:,i] = image[:,:,i] / 255.0
    return normed_image

if __name__ =="__main__":
    im_path = r'H:\Undergraduate\18-19-3\Undergraduate Thesis\Foggy photos\IMG_1716.JPG'
    im = cv2.imread(im_path)
    im = cv2.resize(im, (600, 400), interpolation = cv2.INTER_AREA)
    im_n = norm_image(im)
    cv2.imshow('image0', im)
    #cv2.imshow('image1', im_n)
    cv2.waitKey(0)

