import numpy as np


def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''
# redefine
def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0) # use pad instead of append
    eroded_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            if np.sum(region * kernel) == kernel_ones_count * 255:
                eroded_img[i, j] = 255 # use 255 instead of 1

    return eroded_img


def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0) # use pad instead of append
    dilated_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            if np.any(region * kernel):
                dilated_img[i, j] = 255 # use 255 instead of 1

    return dilated_img

def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)

def hitmiss(img, kernel):
    # A eroded by B
    eroded_img = erode(img, kernel)
    # Ac eroded by Bc
    eroded_complement = erode(255 - img, 1 - kernel)    
    # Hit-or-Miss transform
    hit_or_miss_result = np.logical_and(eroded_img == 255, eroded_complement == 255) * 255    

    return hit_or_miss_result.astype(np.uint8)

# Zhang-Suen thinning algorithm, ref: https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
def thinning(img):
    # convert to binary
    img = img // 255
    
    def neighbours(x, y, img):
        # P9, P2, P3
        # P8, P1, P4
        # P7, P6, P5
        
        # Get P2, P3, P4, P5, P6, P7, P8, P9
        return [img[y-1][x],  img[y-1][x+1],   img[y][x+1],  img[y+1][x+1],  # P2,P3,P4,P5
                img[y+1][x], img[y+1][x-1], img[y][x-1], img[y-1][x-1]]  # P6,P7,P8,P9
    
    def transitions(neighbours):
        # the number of transitions from white to black, (0 -> 1) in the sequence P2,P3,P4,P5,P6,P7,P8,P9, P2.
        n = neighbours + neighbours[0:1] # P2-9 + P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n[:-1], n[1:]))
    
    rows, columns = img.shape
    changing1 = changing2 = [1] # temp
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, rows - 1):
            for x in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = nbs = neighbours(x, y, img)
                if (img[y][x] == 1 and        # 0: The pixel is black and has eight neighbours
                    2 <= sum(nbs) <= 6 and    # 1: 2 <= B(P1) <= 6
                    transitions(nbs) == 1 and # 2: A(P1) = 1
                    P2 * P4 * P6 == 0 and     # 3: At least one of P2/P4/P6 is white
                    P4 * P6 * P8 == 0):       # 4: At least one of P4/P6/P8 is white     
                    
                    changing1.append((x,y))
        for x, y in changing1: img[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, rows - 1):
            for x in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = nbs = neighbours(x, y, img)
                if (img[y][x] == 1 and        # 0: Same as above
                    2 <= sum(nbs) <= 6 and    # 1: Same as above
                    transitions(nbs) == 1 and # 2: Same as above
                    P2 * P4 * P8 == 0 and     # 3: At least one of P2/P4/P8 is white
                    P2 * P6 * P8 == 0):       # 4: At least one of P2/P6/P8 is white     
                    
                    changing2.append((x,y))
        for x, y in changing2: img[y][x] = 0
    return img * 255