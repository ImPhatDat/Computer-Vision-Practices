import sys
import os
import getopt
import cv2
import numpy as np
from morphological_operator import binary

def operator(in_file, out_file, mor_op, wait_key_time=0):
    # img_origin = cv2.imread(in_file)
    # cv2.imshow('original image', img_origin)
    # cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    # cv2.imshow('gray image', img_gray)
    # cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
        
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
        
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 

        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual
        
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
        
    elif mor_op == 'hitmiss':
        img_hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel) 
        cv2.imshow('OpenCV hitmiss image', img_hitmiss)
        cv2.waitKey(wait_key_time)

        img_hitmiss_manual = binary.hitmiss(img, kernel)
        cv2.imshow('manual hitmiss image', img_hitmiss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hitmiss_manual
        
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img) # use Zhang-Suen in default
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thinning(img, kernel)
        cv2.imshow('manual thinning (morph) image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
    elif mor_op == 'zhang_suen':
        img_thinning = cv2.ximgproc.thinning(img) # use Zhang-Suen in default
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thinning(img)
        cv2.imshow('manual thinning (zhang-suen) image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
    
    
    # ------ BONUS ------
    elif mor_op == 'boundary':
        img_boundary_manual = binary.boundary(img, kernel)
        cv2.imshow('manual boundary image', img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual
        
    elif mor_op == 'hole_filling':
        img_fill_hole_manual = binary.fill_hole(img, kernel, xpos=370, ypos=100)
        cv2.imshow('manual fill_hole image', img_fill_hole_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_fill_hole_manual
        
    elif mor_op == 'ccs':
        # Find connected components
        _, labels_im = cv2.connectedComponents(img)
        # Map component labels to hue values for visualization
        label_hue = np.uint8(179 * labels_im / np.max(labels_im))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR) # for display
        labeled_img[label_hue == 0] = 0 # Set background label to black
        cv2.imshow('OpenCV ccs image', labeled_img)
        cv2.waitKey(wait_key_time)
        
        img_ccs_manual = binary.connected_components(img, kernel, xpos=340, ypos=190)
        cv2.imshow('manual ccs image', img_ccs_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_ccs_manual
        
    elif mor_op == 'convex_hull':
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_hull = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # draw contours and convex hulls
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(img_with_hull, [hull], -1, (0, 255, 0), 2)  # Draw green convex hulls
        cv2.imshow('OpenCV convex_hull image', img_with_hull)
        cv2.waitKey(wait_key_time)

        img_convex_hull_manual = binary.convex_hull(img, kernel)
        cv2.imshow('manual convex_hull image', img_convex_hull_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_convex_hull_manual

    

    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
