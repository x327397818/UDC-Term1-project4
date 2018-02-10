# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:51:53 2018

@author: benbe
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

from IPython.display import HTML
#import imageio
##
#imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
"""
# read in and make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

#img_t = mpimg.imread('./camera_cal/calibration1.jpg')
#plt.imshow(img_t)

#arrays to store object points and image points from all the images
objpoints = [] #3D points in the real world
imgpoints = [] #2D points in image

#Prepare object points
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#show chessboard corners
fig_1, axs_1 = plt.subplots(4,5, figsize=(32, 18))
axs_1 = axs_1.ravel()

i = 0
ret_result = []
for fname in images:
    #read in each image
    img = mpimg.imread(fname)

    #convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    ret_result.append(ret)
    #if corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        #draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        axs_1[i].axis('off')
        axs_1[i].imshow(img)
    i += 1
print(ret_result)
fig_1.savefig('./output_images/chessboardimg.jpg')

#Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# save Calibration result
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration.p", "wb" ))

#show sample undistort image
fig_2, axs_2 = plt.subplots(1, 2, figsize=(32,18))
axs_2 = axs_2.ravel()

sample_img = cv2.imread('./camera_cal/calibration1.jpg')
undistort_sample_img = cv2.undistort(sample_img, mtx, dist, None, mtx)

axs_2[0].axis('off')
axs_2[0].set_title('Original Image')
axs_2[0].imshow(sample_img)

axs_2[1].axis('off')
axs_2[1].set_title('Undistorted Image')
axs_2[1].imshow(undistort_sample_img)

fig_2.savefig('./output_images/UndistortedChessBoard.jpg')

fig_3, axs_3 = plt.subplots(1, 2, figsize=(32,18))
axs_3 = axs_3.ravel()
Sample_img_2 = cv2.cvtColor(cv2.imread('./test_images/test1.jpg'), cv2.COLOR_BGR2RGB)

undistort_sample_img_2 = cv2.undistort(Sample_img_2, mtx, dist, None, mtx)

axs_3[0].axis('off')
axs_3[0].set_title('Original Image')
axs_3[0].imshow(Sample_img_2)

axs_3[1].axis('off')
axs_3[1].set_title('Undistorted Image')
axs_3[1].imshow(undistort_sample_img_2)

fig_3.savefig('./output_images/UndistortedSampleImage.jpg')
"""
#test load
dist_pickle = pickle.load(open( "calibration.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

Test_img = cv2.cvtColor(cv2.imread('./test_images/test6.jpg'), cv2.COLOR_BGR2RGB)
undistorted_img = cv2.undistort(Test_img, mtx, dist, None, mtx)

'''
fig, axs = plt.subplots(1, 2, figsize=(32,18))
fig.tight_layout()

axs[0].imshow(undistorted_img)
axs[0].set_title('Undistorted Image')

#draw wrap area
x = [src[0][0],src[2][0],src[3][0],src[1][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1]]
#print(np.array([x,y]).T)
polygon = plt.Polygon(np.array([x,y]).T, color= 'r', linewidth=3,fill =False)
axs[0].add_patch(polygon)

axs[1].imshow(warped_img)
axs[1].set_title('Warped Image')

fig.savefig('./output_images/WarpedTestImage.jpg')
'''
#Filter Channel R 
def RChannel_thresh(img, thresh_min= 0, thresj_max = 255):
    R = img[:,:,0]
    binary = np.zeros_like(R)
    binary[(R > thresh_min) & (R <= thresj_max)] = 1
    
    return binary

def SChannel_thresh(img, thresh_min= 0, thresj_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh_min) & (S <= thresj_max)] = 1
    
    return binary


'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(H)
axs_1[1].set_title('Channel H')

axs_1[2].imshow(L)
axs_1[2].set_title('Channel L')

axs_1[3].imshow(S)
axs_1[3].set_title('Channel S')

fig_1.savefig('./output_images/H-L-S.jpg')
'''

#R_filtered_binary = RChannel_thresh(undistorted_img, thresh_min= 220, thresj_max = 255)
#
#S_filtered_binary = SChannel_thresh(undistorted_img, thresh_min= 120, thresj_max = 255)
#
#S_R_Combined = np.zeros_like(R_filtered_binary)
#
#S_R_Combined[(R_filtered_binary == 1) | (S_filtered_binary == 1)] = 1

'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(S_filtered_binary, cmap='gray')
axs_1[1].set_title('S_filtered_binary')

axs_1[2].imshow(R_filtered_binary, cmap='gray')
axs_1[2].set_title('R_filtered_binary')

axs_1[3].imshow(S_R_Combined, cmap='gray')
axs_1[3].set_title('S_R_Combined')

fig_1.savefig('./output_images/S_R_Combined.jpg')
'''

'''
fig_1,axs_1 = plt.subplots(1, 2, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(R_filtered_binary)
axs_1[1].set_title('R_filtered_binary')

fig_1.savefig('./output_images/R_filtered_binary.jpg')
'''

'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(R)
axs_1[1].set_title('Channel R')

axs_1[2].imshow(G)
axs_1[2].set_title('Channel G')

axs_1[3].imshow(B)
axs_1[3].set_title('Channel B')

fig_1.savefig('./output_images/R-G-B.jpg')
'''

#apply sobel to the img
def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

#gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
#grad_binary_x = abs_sobel_thresh(gray, orient='x', thresh_min=5, thresh_max=230)
#grad_binary_y = abs_sobel_thresh(gray, orient='y', thresh_min=10, thresh_max=230)
#
#Combined_grad_binary = np.zeros_like(gray)
#
#Combined_grad_binary[(grad_binary_x == 1) & (grad_binary_y == 1)] = 1
'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(grad_binary_x, cmap='gray')
axs_1[1].set_title('Thresholded Gradient_X')

axs_1[2].imshow(grad_binary_y, cmap='gray')
axs_1[2].set_title('Thresholded Gradient_Y')

axs_1[3].imshow(grad_binary_y, cmap='gray')
axs_1[3].set_title('Thresholded Gradient')

fig_1.savefig('./output_images/Thresholded Gradient.jpg')
'''

# Calculate gradient magnitude
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # Return the result
    return binary_output

#mag_binary = mag_thresh(gray, sobel_kernel=3, mag_thresh=(5, 255))

'''
fig_1,axs_1 = plt.subplots(1, 2, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(mag_binary, cmap='gray')
axs_1[1].set_title('Thresholded Magnitude')


fig_1.savefig('./output_images/Thresholded Magnitude.jpg')
'''

# Calculate gradient direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    # Return the result
    return dir_binary

#dir_binary = dir_threshold(gray, sobel_kernel=3, thresh=(0.7, 1.4))

'''
fig_1,axs_1 = plt.subplots(1, 2, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(dir_binary, cmap='gray')
axs_1[1].set_title('Thresholded Grad. Dir.')


fig_1.savefig('./output_images/Thresholded Grad. Dir.jpg')
'''
#combined_mag_dir = np.zeros_like(gray)
#
#combined_mag_dir[(mag_binary == 1) & (dir_binary == 1)] = 1
'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(undistorted_img)
axs_1[0].set_title('undistorted Image')

axs_1[1].imshow(mag_binary, cmap='gray')
axs_1[1].set_title('Thresholded Magnitude')

axs_1[2].imshow(dir_binary, cmap='gray')
axs_1[2].set_title('Thresholded Grad. Dir.')

axs_1[3].imshow(combined_mag_dir, cmap='gray')
axs_1[3].set_title('combined Mag Dir')


fig_1.savefig('./output_images/combined Mag Dir.jpg')
'''
#Combined_img = np.zeros_like(gray)
#
#Combined_img[((S_R_Combined == 1) | ((Combined_grad_binary == 1) & (combined_mag_dir == 1)))] = 1
'''
fig_1,axs_1 = plt.subplots(1, 4, figsize=(32,18))
fig_1.tight_layout()

axs_1[0].imshow(Combined_img, cmap='gray')
axs_1[0].set_title('Combined_img')

axs_1[1].imshow(S_R_Combined, cmap='gray')
axs_1[1].set_title('S_R_Combined')

axs_1[2].imshow(Combined_grad_binary, cmap='gray')
axs_1[2].set_title('Combined_grad_binary')

axs_1[3].imshow(combined_mag_dir, cmap='gray')
axs_1[3].set_title('combined_mag_dir')



fig_1.savefig('./output_images/Combined_img.jpg')
'''

def warp_img(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

h,w = Test_img.shape[:2]

offset = 200

src = np.float32([(580,460),
                  (735,460), 
                  (250,675), 
                  (1130,675)])
    
dst = np.float32([(offset,0),
                  (w-offset,0),
                  (offset,h),
                  (w-offset,h)])

#warped_img, M, Minv = warp_img(S_R_Combined, src, dst)
#warped_img_1, M, Minv = warp_img(Combined_img, src, dst)
'''
fig, axs = plt.subplots(1, 2, figsize=(32,18))
fig.tight_layout()

axs[0].imshow(S_R_Combined, cmap='gray')
axs[0].set_title('S_R_Combined')

#draw wrap area
x = [src[0][0],src[2][0],src[3][0],src[1][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1]]
#print(np.array([x,y]).T)
polygon = plt.Polygon(np.array([x,y]).T, color= 'r', linewidth=3,fill =False)
axs[0].add_patch(polygon)

axs[1].imshow(warped_img, cmap='gray')
axs[1].set_title('Warped Image')

#fig.savefig('./output_images/WarpedImage_1.jpg')
'''

margin = 80

def find_lane(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)

    
    leftx_base = np.argmax(histogram[200:midpoint]) +200
    rightx_base = np.argmax(histogram[midpoint:1200]) + midpoint
    
    print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    #margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

#left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = find_lane(warped_img)

'''
histogram = visualization_data[1]
## Print histogram from sliding window polyfit for example image
fig, axs = plt.subplots(1, 1, figsize=(32,18))
axs.plot(histogram)
#axs.xlim(0, 1280)
fig.savefig('./output_images/Historgram.jpg')

h = warped_img.shape[0]
left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
#print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

rectangles = visualization_data[0]
histogram = visualization_data[1]

# Create an output image to draw on and  visualize the result
out_img = np.uint8(np.dstack((warped_img, warped_img, warped_img))*255)
# Generate x and y values for plotting
ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
for rect in rectangles:
# Draw the windows on the visualization image
    cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
    cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
fig, axs = plt.subplots(1, 1, figsize=(32,18))
axs.imshow(out_img)
axs.plot(left_fitx, ploty, color='yellow')
axs.plot(right_fitx, ploty, color='yellow')
#axs.xlim(0, 1280)
#axs.ylim(720, 0)

fig.savefig('./output_images/SlideWidows.jpg')

'''
def fit_use_prev(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds



#left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = fit_use_prev(warped_img_1, left_fit, right_fit)

'''
# Generate x and y values for plotting
ploty = np.linspace(0, warped_img_1.shape[0]-1, warped_img_1.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
left_fitx2 = left_fit2[0]*ploty**2 + left_fit2[1]*ploty + left_fit2[2]
right_fitx2 = right_fit2[0]*ploty**2 + right_fit2[1]*ploty + right_fit2[2]

# Create an image to draw on and an image to show the selection window
out_img = np.uint8(np.dstack((warped_img_1, warped_img_1, warped_img_1))*255)
window_img = np.zeros_like(out_img)

# Color in left and right line pixels
nonzero = warped_img_1.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area (OLD FIT)
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
fig, axs = plt.subplots(1, 1, figsize=(32,18))
axs.imshow(result)
axs.plot(left_fitx2, ploty, color='yellow')
axs.plot(right_fitx2, ploty, color='yellow')

fig.savefig('./output_images/fit_use_prev.jpg')
'''

def curv_rad_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
    
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist


#rad_l, rad_r, d_center = curv_rad_center_dist(warped_img, left_fit, right_fit, left_lane_inds, right_lane_inds)

'''
print('Radius of curvature for example:', rad_l, 'm,', rad_r, 'm')
print('Distance from lane center for example:', d_center, 'm')
'''
def draw_lane(original_img, binary_img, l_fit, r_fit, Minv, curv_rad, center_dist):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(result, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(result, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    
    return result
'''
DrawLaneImg = draw_lane(Test_img, warped_img, left_fit, right_fit, Minv, (rad_l+rad_r)/2, d_center)

fig, axs = plt.subplots(1, 1, figsize=(32,18))
axs.imshow(DrawLaneImg)
fig.savefig('./output_images/draw_lane.jpg')
'''
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False 
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
        
    def add_fit(self, fit, inds):
        self.diffs = np.array([0,0,0], dtype='float')
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        
        else:
            self.detected = False
#            if len(self.current_fit) > 1:
#                # throw out oldest fit
#                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)
                
def pipeline(img):
    # Undistort
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    R_filtered_binary = RChannel_thresh(undistorted_img, thresh_min= 180, thresj_max = 255)

    S_filtered_binary = SChannel_thresh(undistorted_img, thresh_min= 70, thresj_max = 255)

    S_R_Combined = np.zeros_like(R_filtered_binary)

    S_R_Combined[(R_filtered_binary == 1) | (S_filtered_binary == 1)] = 1
    
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
    grad_binary_x = abs_sobel_thresh(gray, orient='x', thresh_min=5, thresh_max=230)
    grad_binary_y = abs_sobel_thresh(gray, orient='y', thresh_min=10, thresh_max=230)

    Combined_grad_binary = np.zeros_like(gray)

    Combined_grad_binary[(grad_binary_x == 1) & (grad_binary_y == 1)] = 1
    
    mag_binary = mag_thresh(gray, sobel_kernel=3, mag_thresh=(5, 255))
    
    dir_binary = dir_threshold(gray, sobel_kernel=3, thresh=(0.7, 1.4))
    
    combined_mag_dir = np.zeros_like(gray)

    combined_mag_dir[(mag_binary == 1) & (dir_binary == 1)] = 1
    
    Combined_img = np.zeros_like(gray)

    Combined_img[((S_R_Combined == 1) | ((Combined_grad_binary == 1) & (combined_mag_dir == 1)))] = 1
    
    # Perspective Transform
    warpimg, M, Minv = warp_img(S_R_Combined, src, dst)
#    plt.imshow(warpimg)
    return warpimg, Minv

#warped_img, Minv = pipeline(Test_img)

def process_image(img):
    #new_img = np.copy(img)
    processed_img, Minv = pipeline(img)
    
    # if both left and right lines were detected last frame, use fit_use_prev, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, _ = find_lane(processed_img)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = fit_use_prev(processed_img, l_line.best_fit, r_line.best_fit)
        
    # Filter the finded lines
    if l_fit is not None and r_fit is not None:
        
        h = img.shape[0]
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(650 - x_int_diff) > 100:
            l_fit = None
            r_fit = None
            
    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)
    
    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        rad_l, rad_r, d_center = curv_rad_center_dist(processed_img, l_line.best_fit, r_line.best_fit, l_lane_inds, r_lane_inds)
        img_out = draw_lane(img, processed_img, l_line.best_fit, r_line.best_fit, Minv, (rad_l+rad_r)/2, d_center)
    else:
        img_out = img
        
#    diagnostic_output = True
#    if diagnostic_output:
#        # put together multi-view output
#        diag_img = np.zeros((720,1280,3), dtype=np.uint8)
#        
#        # original output (top left)
#        diag_img[0:360,0:640,:] = cv2.resize(img_out,(640,360))
#        
#        # binary overhead view (top right)
#        processed_img = np.dstack((processed_img*255, processed_img*255, processed_img*255))
#        resized_img_bin = cv2.resize(processed_img,(640,360))
#        diag_img[0:360,640:1280, :] = resized_img_bin
#        
#        # overhead with all fits added (bottom right)
#        img_bin_fit = np.copy(processed_img)
#        for i, fit in enumerate(l_line.current_fit):
#            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))
#        for i, fit in enumerate(r_line.current_fit):
#            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0,20*i+100,20*i+100))
#        img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255,255,0))
#        img_bin_fit = plot_fit_onto_img(img_bin_fit, r_line.best_fit, (255,255,0))
#        diag_img[360:720,640:1280,:] = cv2.resize(img_bin_fit,(640,360))
#        
#        # diagnostic data (bottom left)
#        color_ok = (200,255,155)
#        color_bad = (255,155,155)
#        font = cv2.FONT_HERSHEY_DUPLEX
#        if l_fit is not None:
#            text = 'This fit L: ' + ' {:0.6f}'.format(l_fit[0]) + \
#                                    ' {:0.6f}'.format(l_fit[1]) + \
#                                    ' {:0.6f}'.format(l_fit[2])
#        else:
#            text = 'This fit L: None'
#        cv2.putText(diag_img, text, (40,380), font, .5, color_ok, 1, cv2.LINE_AA)
#        if r_fit is not None:
#            text = 'This fit R: ' + ' {:0.6f}'.format(r_fit[0]) + \
#                                    ' {:0.6f}'.format(r_fit[1]) + \
#                                    ' {:0.6f}'.format(r_fit[2])
#        else:
#            text = 'This fit R: None'
#        cv2.putText(diag_img, text, (40,400), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Best fit L: ' + ' {:0.6f}'.format(l_line.best_fit[0]) + \
#                                ' {:0.6f}'.format(l_line.best_fit[1]) + \
#                                ' {:0.6f}'.format(l_line.best_fit[2])
#        cv2.putText(diag_img, text, (40,440), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Best fit R: ' + ' {:0.6f}'.format(r_line.best_fit[0]) + \
#                                ' {:0.6f}'.format(r_line.best_fit[1]) + \
#                                ' {:0.6f}'.format(r_line.best_fit[2])
#        cv2.putText(diag_img, text, (40,460), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Diffs L: ' + ' {:0.6f}'.format(l_line.diffs[0]) + \
#                             ' {:0.6f}'.format(l_line.diffs[1]) + \
#                             ' {:0.6f}'.format(l_line.diffs[2])
#        if l_line.diffs[0] > 0.001 or \
#           l_line.diffs[1] > 1.0 or \
#           l_line.diffs[2] > 100.:
#            diffs_color = color_bad
#        else:
#            diffs_color = color_ok
#        cv2.putText(diag_img, text, (40,500), font, .5, diffs_color, 1, cv2.LINE_AA)
#        text = 'Diffs R: ' + ' {:0.6f}'.format(r_line.diffs[0]) + \
#                             ' {:0.6f}'.format(r_line.diffs[1]) + \
#                             ' {:0.6f}'.format(r_line.diffs[2])
#        if r_line.diffs[0] > 0.001 or \
#           r_line.diffs[1] > 1.0 or \
#           r_line.diffs[2] > 100.:
#            diffs_color = color_bad
#        else:
#            diffs_color = color_ok
#        cv2.putText(diag_img, text, (40,520), font, .5, diffs_color, 1, cv2.LINE_AA)
#        text = 'Good fit count L:' + str(len(l_line.current_fit))
#        cv2.putText(diag_img, text, (40,560), font, .5, color_ok, 1, cv2.LINE_AA)
#        text = 'Good fit count R:' + str(len(r_line.current_fit))
#        cv2.putText(diag_img, text, (40,580), font, .5, color_ok, 1, cv2.LINE_AA)
#        
#        img_out = diag_img
    
    return img_out

def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    #new_img = np.copy(img)
    h = img.shape[0]
    ploty = np.linspace(0, h-1, h)
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return img

l_line = Line()
r_line = Line()

#video_output = 'project_video_output.mp4'
#video_input = VideoFileClip('project_video.mp4')
#processed_video = video_input.fl_image(process_image)
#processed_video.write_videofile(video_output, audio=False)

video_output1 = 'challenge_video_output.mp4'
video_input1 = VideoFileClip('challenge_video.mp4')
processed_video = video_input1.fl_image(process_image)
processed_video.write_videofile(video_output1, audio=False)

#video_output2 = 'harder_challenge_video_output.mp4'
#video_input2 = VideoFileClip('harder_challenge_video.mp4')
#processed_video = video_input2.fl_image(process_image)
#processed_video.write_videofile(video_output2, audio=False)