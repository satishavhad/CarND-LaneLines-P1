# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:07:34 2017

@author: avhadsa
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    left_slope=[]
    left_lines=[]
    right_slope=[]
    right_lines=[]
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y2-y1)/(x2-x1)) # slope
            #print (m)
            if m >= 0.5 and m<=0.8 :
                left_slope.append(m)
                left_lines.append((x1,y1))
                #print ("left slope is", left_slope)
            elif m <= -0.6 and m >= -0.88:
                right_slope.append(m)
                right_lines.append((x2,y2))
                #print ("right slope is", right_slope)

    # average left and right slopes
    right_slope = sorted(right_slope)[int(len(right_slope)/2)]
    left_slope = sorted(left_slope)[int(len(left_slope)/2)]
    
    start_left_y = sorted([line[1] for line in left_lines])[int(len(left_lines)/2)]
    start_left_x = [line[0] for line in left_lines if line[1] == start_left_y][0]

    start_right_y = sorted([line[1] for line in right_lines])[int(len(right_lines)/2)]
    start_right_x = [line[0] for line in right_lines if line[1] == start_right_y][0]

    # next we extend to the car
    end_left_x = int((img.shape[1]-start_left_y)/left_slope) + start_left_x
    end_right_x = int((img.shape[1]-start_right_y)/right_slope) + start_right_x
                

    b1_left=start_left_y - (left_slope * start_left_x  )             
    b2_right=start_right_y - (right_slope *start_right_x )
    print(start_left_x,start_left_y,b1_left,start_right_x,start_right_y,b2_right)
    
    #if start_left_y>350:
    start_left_y=350
    start_left_x=int((start_left_y-b1_left)/left_slope)
    
    #if start_right_y>350:
    start_right_y=350
    start_right_x=int((start_right_y-b2_right)/right_slope)
                       
    cv2.line(img, (start_left_x, start_left_y), (end_left_x, img.shape[1]), color, thickness)
    cv2.line(img, (start_right_x, start_right_y), (end_right_x, img.shape[1]), color, thickness)            
            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(img):    
    #reading in an image
    image = mpimg.imread(img)
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    
    
    gray = grayscale(image)
    plt.imshow(gray, cmap='gray')
    
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    plt.imshow(edges, cmap='Greys_r')
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(440, 300), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    plt.imshow(masked_edges, cmap='Greys_r')
    
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 #minimum number of pixels making up a line
    max_line_gap = 160    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    plt.imshow(line_image)
    
    lines_edges = weighted_img(line_image,image)
    plt.imshow(lines_edges)
    
    #cv2.imwrite("test_images/solidYellowLeft.jpg",lines_edges) 


import os
All_images=os.listdir("test_images/")

process_image("test_images/whiteCarLaneSwitch.jpg")

#solidWhiteCurve
#solidWhiteRight
#solidYellowCurve
#solidYellowCurve2
#solidYellowLeft
#whiteCarLaneSwitch

#for image in All_images:
#    process_image("test_images/"+image)



