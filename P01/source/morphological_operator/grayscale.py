import numpy as np
from cv2 import imshow, waitKey # for debugging and view process only
import mahotas # get structuring elements

def disk(radius):
    return mahotas.morph.disk(radius).astype(np.uint8)

def erode(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=255) # since we use min here
    eroded_img = np.zeros_like(img)
    foreground = kernel == 1
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            eroded_img[i, j] = np.min(region[foreground])
            
    if n == 1:
        return eroded_img
    else:
        return erode(eroded_img, kernel, n - 1)
    
def dilate(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0) # since we use max here
    dilated_img = np.zeros_like(img)
    foreground = kernel == 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            dilated_img[i, j] = np.max(region[foreground])
            
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

def granulometry(img, max_size=35):
    img = smoothing(img, disk(5))
    imshow(f"Smoothed image", img)
    waitKey(100)
    
    area_diffs = []
    initial_area = np.sum(img, dtype=np.uint64) # initial
    
    for size in range(2, max_size + 1):
        kernel = disk(size)
        
        # Apply opening operation
        opened_image = opening(img, kernel)
        if size % 5 == 0:
            imshow(f"Opened with kernel size of {size}", opened_image)
            waitKey(100)
        opened_area = np.sum(opened_image, dtype=np.uint64)
        
        # Calculate the area difference
        area_diff = np.subtract(np.uint64(initial_area), np.uint64(opened_area))
        
        area_diffs.append(area_diff)
        
        initial_area = opened_area
    
    return area_diffs
    
def textural_segmentation(img, kernel):
    closed = closing(img, disk(30))
    imshow("Closed image", closed)
    waitKey(500)
    opened = opening(closed, disk(60))
    imshow("Opened image", opened)
    waitKey(500)
    boundary = gradient(opened, kernel)
    imshow("Boundary (obtained by gradient)", boundary)
    waitKey(500)
    
    # impose on img
    return boundary + img


# -- RECONSTRUCTION --

def pointwise_min(img1, img2):
    return np.minimum(img1, img2) # same as intersect

def pointwise_max(img1, img2):
    return np.maximum(img1, img2) # same as union

def geodesic_dilate(marker, mask, kernel):
    return pointwise_min(dilate(marker, kernel), mask)

def geodesic_erode(marker, mask, kernel):
    return pointwise_max(erode(marker, kernel), mask)

def reconstruction_dilate(marker, mask, kernel):
    current = geodesic_dilate(marker, mask, kernel)
    while True:
        prev = current
        current = geodesic_dilate(prev, mask, kernel)
        
        imshow(f"in progress", current)
        waitKey(50)
        
        if np.array_equal(prev, current):
            break
    return current

def reconstruction_erode(marker, mask, kernel):
    current = geodesic_erode(marker, mask, kernel)
    while True:
        prev = current
        current = geodesic_erode(prev, mask, kernel)
        
        imshow(f"in progress", current)
        waitKey(50)
        
        if np.array_equal(prev, current):
            break
    return current

def reconstruction_opening(img, kernel, n=1):
    img_erode = erode(img, kernel, n)
    imshow(f'Eroded image {n} times', img_erode)
    waitKey(500)
    img_recon_dilate_manual = reconstruction_dilate(img_erode, img, kernel)
    return img_recon_dilate_manual

def reconstruction_closing(img, kernel, n=1):
    img_dilate = dilate(img, kernel, n)
    imshow(f'Dilated image {n} times', img_dilate)
    waitKey(500)
    img_recon_dilate_manual = reconstruction_erode(img_dilate, img, kernel)
    return img_recon_dilate_manual

def reconstruction_tophat(img, kernel, n=1):
    return subtract(img, reconstruction_opening(img, kernel, n))

