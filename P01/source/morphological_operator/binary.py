import numpy as np
from cv2 import imshow, waitKey # for debugging and view process only

# def erode(img, kernel):
#     kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
#     kernel_ones_count = kernel.sum()
#     eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
#     img_shape = img.shape

#     x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
#     img = np.append(img, x_append, axis=1)

#     y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
#     img = np.append(img, y_append, axis=0)

#     for i in range(img_shape[0]):
#         for j in range(img_shape[1]):
#             i_ = i + kernel.shape[0]
#             j_ = j + kernel.shape[1]
#             if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
#                 eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

#     return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''
# helpers
def complement(img):
    return 255 - img

def intersect_image(img1, img2):
    return np.minimum(img1, img2)

def union_image(img1, img2):
    return np.maximum(img1, img2)

def subtract(img1, img2):
    return (np.clip(img1 - img2, 0, 255)).astype(np.uint8)

# redefine
def erode(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0)
    eroded_img = np.zeros_like(img)
    foreground = kernel == 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            if np.all(region[foreground] == 255):
                eroded_img[i, j] = 255 # use 255 instead

    if n == 1:
        return eroded_img
    else:
        return erode(eroded_img, kernel, n - 1)

def dilate(img, kernel, n=1):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0)
    dilated_img = np.zeros_like(img)
    foreground = kernel == 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            if np.any(region[foreground] == 255):
                dilated_img[i, j] = 255 # use 255 instead

    if n == 1:
        return dilated_img
    else:
        return dilate(dilated_img, kernel, n - 1)

def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)

# Hit-or-Miss transformation
def hitmiss(img, kernel):
    # # A eroded by B
    # eroded_img = erode(img, kernel)
    # # Ac eroded by Bc
    # eroded_complement = erode(complement(img), -1 * kernel)   
    # # intersect
    # return intersect_image(eroded_img, eroded_complement)
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    padded_img = np.pad(img, ((kernel_center[0], kernel_center[0]), (kernel_center[1], kernel_center[1])), 
                        mode='constant', constant_values=0)
    eroded_img = np.zeros_like(img)
    foreground = kernel == 1
    background = kernel == -1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            if np.all(region[foreground] == 255) and np.all(region[background] == 0):
                eroded_img[i, j] = 255 # use 255 instead

    return eroded_img 

def simplified_hitmiss(img, kernel):
    # erode without background match requirement
    return erode(img, kernel)

# Zhang-Suen thinning algorithm
def zhangsuen_thinning(img):
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

def thinning(img):
    def single_thinning(img, kernel):
        # use morphological operators: A - (A hitmiss B)
        # or A intersect (A hitmiss B)c
        return subtract(img, hitmiss(img, kernel))
    # structuring elements
    Bs = [None for _ in range(8)]
    Bs[0] = np.array([[-1, -1, -1],
                    [0, 1, 0],
                    [1, 1, 1]], dtype=np.int8) # B1
    Bs[1] = np.array([[0, -1, -1],
                    [1, 1, -1],
                    [1, 1, 0]], dtype=np.int8) # B2
    Bs[2], Bs[4], Bs[6] = [np.rot90(Bs[0], k + 1, axes=(1,0)) for k in range(3)]
    Bs[3], Bs[5], Bs[7] = [np.rot90(Bs[0], k + 1, axes=(1,0)) for k in range(3)]
    
    current = img
    while True:
        prev = current.copy()
        current = prev
        for B in Bs:
            current = single_thinning(current, B)
        
        imshow("in progress", current)
        waitKey(50)
        
        if np.array_equal(prev, current):
            break
    return current


# ------ BONUS ------

# Boundary Extraction
def boundary(img, kernel):
    # A - (A eroded by B)
    return subtract(img, erode(img, kernel))

# Hole Filling
def fill_hole(img, kernel, xpos, ypos):
    # Xk = (X{k - 1} dilated by B) intersect Ac
    imgc = complement(img)
    X_current = np.zeros_like(img)
    X_current[ypos, xpos] = 255
    while True: 
        X_prev = X_current.copy()
        X_current = intersect_image(dilate(X_prev, kernel), imgc)
        
        imshow("in progress", X_current)
        waitKey(50)
        
        # loop untils Xk = X{k - 1}
        if np.array_equal(X_prev, X_current):
            break
    res = union_image(X_current, img)
    return res

# Extraction of Connected Components
def connected_components(img, kernel, xpos, ypos):
    # Xk = (X{k - 1} dilated by B) intersect A
    X_current = np.zeros_like(img)
    X_current[ypos, xpos] = 255
    while True: 
        X_prev = X_current.copy()
        X_current = intersect_image(dilate(X_prev, kernel), img)
        
        imshow("in progress", X_current)
        waitKey(50)
        
        # loop untils Xk = X{k - 1}
        if np.array_equal(X_prev, X_current):
            break
    return X_current

# Convex Hull
def convex_hull(img):
    # default using left, right, top, bottom borders
    tmp_kernel = np.full((3, 3), 0)
    tmp_kernel[1, 1] = -1
    Bs = [tmp_kernel.copy() for _ in range(4)]
    
    Bs[0][:, 0] = 1 # left
    Bs[1][0, :] = 1 # top
    Bs[2][:, -1] = 1 # right
    Bs[3][-1, :] = 1 # bottom
    
    X_currents = [img.copy() for _ in range(len(Bs))]
    
    for i in range(len(X_currents)):
        while True:
            imshow(f"in progress ({i})", X_currents[i])
            waitKey(1000)
            
            X_prev = X_currents[i].copy()
            X_currents[i] = union_image(simplified_hitmiss(X_currents[i], Bs[i]), img)
            
            if np.array_equal(X_currents[i], X_prev):
                break
    
    # union D (last X) to get C(A)
    res = X_currents[0]
    for i in range(len(X_currents) - 1):
        res = union_image(res, X_currents[i + 1])
    return res.astype(np.uint8)
    
# Thickening
def thickening(img):
    return complement(thinning(complement(img)))

# Skeletons
def skeletons(img, kernel):
    res = np.zeros_like(img)
    k = 0
    while True:
        kerode = img
        for _ in range(k):
            kerode = erode(kerode, kernel)
        if kerode.sum() == 0:
            break
        
        sk = subtract(kerode, opening(kerode, kernel))
        res = union_image(res, sk)
        
        imshow(f"in progress", res)
        waitKey(50)
        
        k += 1
    return res

# Pruning
def pruning(img, n=3):
    # structuring elements
    Bs = []
    Bs.append(np.array([[0, -1 , -1],
                        [1, 1, -1],
                        [0, -1, -1]], dtype=np.int8)) # B1
    Bs.extend([np.rot90(Bs[0], k + 1, axes=(1,0)) for k in range(3)]) # B2, B3, B4
    Bs.append(np.array([[1, -1 , -1],
                        [-1, 1, -1],
                        [-1, -1, -1]], dtype=np.int8)) # B5
    Bs.extend([np.rot90(Bs[4], k + 1, axes=(1,0)) for k in range(3)]) # B6, B7, B8
    
    # Apply thinning n times: X1
    X1 = img.copy()
    for _ in range(n):
        for B in Bs:
            X1 = thinning(X1, B)
    
    imshow(f"X1 (thinning)", X1)
    waitKey(50)
    
    # Get endpoints: X2
    X2 = simplified_hitmiss(X1, Bs[0])
    for i in range(len(Bs) - 1):
        X2 = union_image(X2, simplified_hitmiss(X1, Bs[i + 1]))
        
    imshow(f"X2 (endpoints)", X2)
    waitKey(50)
        
    # Dilation by H n times
    H = np.ones((3, 3), dtype=np.uint8) # reconstruction dilation
    
    X3 = intersect_image(dilate(X2, H), img)
    for _ in range(n - 1):
        X3 = intersect_image(dilate(X3, H), img)
    
    imshow(f"X3 (reconstruction dilation)", X3)
    waitKey(50)
    
    # Union X1 X3
    return union_image(X1, X3)

# -- Morphological Reconstruction --

def geodesic_dilate(marker, mask, kernel):
    return intersect_image(dilate(marker, kernel), mask)

def geodesic_erode(marker, mask, kernel):
    return union_image(erode(marker, kernel), mask)

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

def reconstruction_opening(img, kernel, n):
    img_erode = erode(img, kernel, n)
    imshow(f'Eroded image {n} times', img_erode)
    waitKey(500)
    img_recon_dilate_manual = reconstruction_dilate(img_erode, img, kernel)
    return img_recon_dilate_manual

def reconstruction_closing(img, kernel, n):
    img_dilate = dilate(img, kernel, n)
    imshow(f'Dilated image {n} times', img_dilate)
    waitKey(500)
    img_recon_dilate_manual = reconstruction_erode(img_dilate, img, kernel)
    return img_recon_dilate_manual

def hole_fill_all(img, kernel):
    F = np.zeros_like(img)
    # set border to 255 (or 1 in binary) - img
    F[0, :] = 255 - img[0, :]
    F[-1, :] = 255 - img[-1, :]
    F[:, 0] = 255 - img[:, 0]
    F[:, -1] = 255 - img[:, -1]
    
    return complement(reconstruction_dilate(F, complement(img), kernel))

def border_clearing(img, kernel):
    F = np.zeros_like(img)
    # set border to border of img
    F[0, :] = img[0, :]
    F[-1, :] = img[-1, :]
    F[:, 0] = img[:, 0]
    F[:, -1] = img[:, -1]
    
    rd = reconstruction_dilate(F, img, kernel)
    return img - rd