import numpy as np
from cv2 import imshow, waitKey # for debugging and view process only


def erode(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0)
    eroded_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            eroded_img[i, j] = np.min(region)
    if n == 1:
        return eroded_img
    else:
        return erode(eroded_img, kernel, n - 1)
    
def dilate(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0)
    dilated_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            dilated_img[i, j] = np.max(region)
    if n == 1:
        return dilated_img
    else:
        return dilate(dilated_img, kernel, n - 1)
    
def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)

def smoothing(img, kernel):
    return closing(opening(img, kernel), kernel)


def subtract(img1, img2):
    return (np.clip(img1 - img2, 0, 255)).astype(np.uint8)

def gradient(img, kernel):
    return subtract(dilate(img, kernel), erode(img, kernel))

def top_hat(img, kernel):
    return subtract(img, opening(img, kernel))

def bottom_hat(img, kernel):
    return subtract(closing(img, kernel), img)