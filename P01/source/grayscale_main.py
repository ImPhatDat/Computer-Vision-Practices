import sys
import os
import getopt
import cv2
import numpy as np
from morphological_operator import grayscale

def operator(in_file, out_file, mor_op, wait_key_time=0):
    img = cv2.imread(in_file, 0)
    cv2.imshow('grayscale image', img)
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

        img_dilation_manual = grayscale.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
        
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = grayscale.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
        
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 

        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = grayscale.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual
        
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = grayscale.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    
    elif mor_op == 'smoothing':
        img_smoothing = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), 
                                            cv2.MORPH_OPEN, 
                                            kernel) 
        cv2.imshow('OpenCV smoothing image', img_smoothing)
        cv2.waitKey(wait_key_time)

        img_smoothing_manual = grayscale.smoothing(img, kernel)
        cv2.imshow('manual smoothing image', img_smoothing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_smoothing_manual
        
    elif mor_op == 'gradient':
        img_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = grayscale.gradient(img, kernel)
        cv2.imshow('manual gradient image', img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual
        
    elif mor_op == 'top_hat':
        img_top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV top_hat image', img_top_hat)
        cv2.waitKey(wait_key_time)

        img_top_hat_manual = grayscale.top_hat(img, kernel)
        cv2.imshow('manual top_hat image', img_top_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_top_hat_manual
        
    elif mor_op == 'bottom_hat':
        img_bottom_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV bottom_hat image', img_bottom_hat)
        cv2.waitKey(wait_key_time)

        img_bottom_hat_manual = grayscale.bottom_hat(img, kernel)
        cv2.imshow('manual bottom_hat image', img_bottom_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_bottom_hat_manual
    
    else:
        print("Not existed operator")
        return

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
    # cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
