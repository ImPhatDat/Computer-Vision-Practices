import sys
import getopt
import cv2
from morphological_operators import grayscale

def operator(in_file, out_file, mor_op, wait_key_time=0):
    img = cv2.imread(in_file, 0)
    cv2.imshow('grayscale image', img)
    cv2.waitKey(wait_key_time)

    kernel = grayscale.disk(3)
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
        # # for demo
        # kernel = np.ones((1, 32), dtype=np.uint8)
        
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
        img_smoothing = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), 
                                            cv2.MORPH_CLOSE, 
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
        
    elif mor_op == 'granulometry':
        smooth_radius = int(input("Enter the disk radius for smoothing: "))
        max_size = int(input("Enter the disk radius max radius: "))
        
        area_diffs = grayscale.granulometry(img, smooth_radius=smooth_radius, max_size=max_size)
        # Plot the granulometry curve
        import matplotlib.pyplot as plt
        plt.plot(range(len(area_diffs)), area_diffs, marker='o')
        plt.title('Granulometry')
        plt.xlabel('r')
        plt.ylabel('Differences in surface area')
        plt.grid(True)
        plt.savefig(out_file)
        plt.show()
        
    elif mor_op == 'textural_segmentation':
        close_radius = int(input("Enter the disk radius for closing: "))
        open_radius = int(input("Enter the disk radius for opening: "))
        
        img_segment_manual = grayscale.textural_segmentation(img, 
                                                             close_radius=close_radius, 
                                                             open_radius=open_radius,
                                                             gradient_kernel=kernel)
        cv2.imshow('manual segment image', img_segment_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_segment_manual
        
    elif mor_op == 'recon_opening':
        # # for demo
        # kernel = np.ones((1, 32), dtype=np.uint8)
        
        img_recon_opening_manual = grayscale.reconstruction_opening(img, kernel, n=1)
        cv2.imshow('manual recon_opening image', img_recon_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_recon_opening_manual
    
    elif mor_op == 'recon_closing':
        # # for demo
        # kernel = np.ones((1, 32), dtype=np.uint8)
        
        img_recon_closing_manual = grayscale.reconstruction_closing(img, kernel, n=1)
        cv2.imshow('manual recon_closing image', img_recon_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_recon_closing_manual
        
    elif mor_op == 'recon_tophat':
        # # for demo
        # kernel = np.ones((1, 32), dtype=np.uint8)
        
        img_recon_tophat_manual = grayscale.reconstruction_tophat(img, kernel, n=1)
        cv2.imshow('manual recon_tophat image', img_recon_tophat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_recon_tophat_manual
        
        
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
