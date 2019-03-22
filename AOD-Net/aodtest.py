import cv2

def resize_image(image, width, height):
    resized_image = image
    if image.shape != (height, width, 3):
        resized_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resized_image
