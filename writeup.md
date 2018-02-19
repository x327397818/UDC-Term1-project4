
**Advanced Lane Finding Project**
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboardimg]: ./output_images/chessboardimg.jpg "chessboardimg"
[UndistortedChessBoard]: ./output_images/UndistortedChessBoard.jpg "Road UndistortedChessBoard"
[UndistortedSampleImage]: ./output_images/UndistortedSampleImage.jpg "UndistortedSampleImage"
[R-G-B]: ./output_images/R-G-B.jpg "R-G-B"
[H-L-S]: ./output_images/H-L-S.jpg "H-L-S"
[S_R_Combined]: ./output_images/S_R_Combined.jpg "S_R_Combined"
[Thresholded Gradient]: ./output_images/Thresholded_Gradient.jpg "Thresholded Gradient"
[combined Mag Dir]: ./output_images/Combined_Mag_Dir.jpg "combined Mag Dir"
[Combined_img_or]: ./output_images/Combined_img_or.jpg "Combined_img_or"
[WarpedImage]: ./output_images/WarpedImage.jpg "WarpedImage"
[Historgram]: ./output_images/Historgram.jpg "Historgram"
[SlideWidows]: ./output_images/SlideWidows.jpg "SlideWidows"
[fit_use_prev]: ./output_images/fit_use_prev.jpg "fit_use_prev"
[draw_lane]: ./output_images/draw_lane.jpg "draw_lane"


[project_video_output]: ./project_video_output.mp4 "project_video_output"
[challenge_video_output]: ./challenge_video_output.mp4 "challenge_video_output"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 18 through 65 of the file called `Adv_lane_detect_clean.py`.  

OpenCV functions findChessboardCorners and calibrateCamera are the key functions in the image calibration. In the `camera_cal` fold, there are 20 images of chessboard as the resource to do calibration.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Generally, these coefficients will not change for a given camera (and lens). The below image shows the corners drawn onto twenty chessboard images using function `drawChessboardCorners()`:

![alt text][chessboardimg]

Some of the chessboard images don't appear because findChessboardCorners was unable to detect the desired number of internal corners.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][UndistortedChessBoard]

Then I save the coefficients to the pickle file `calibration.p`, so if coefficients can be loaded from the file in the future instead of do calibration every time.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][UndistortedSampleImage]

The effect of undistort is hard to see, but can be figured out from the difference in shape of the hood of the car at the bottom of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 77 through 145 in `Adv_lane_detect_clean.py`).  Here's an example of my output for this step.

#### 1) Color filter

The below image shows the RGB and HLS channels of three different color spaces for the same image:
![alt text][R-G-B]
![alt text][H-L-S]

As seen from the channels, R and S channels give the best recognition for the lanes.
So I tune the pixel parameters to filter out the lanes,

The pass for the R channel is `[180,255]`

The pass for the S channel is `[70,255]`

Then I combined them with `OR` operation to filter the image as following image shows,
![alt text][S_R_Combined] 

#### 2) Sobel magnitude and direction

After tuning the parameters, here is result of filters parameters,

Sobel_X `[5,230]`
Sobel_Y `[10,230]`

Magnitude `[5,255]`

direction `[0.7,1.4]`,

First I combined Sobel_X and Sobel_Y with `AND` operation

![alt text][Thresholded Gradient]

Then combine Magnitude and direction with `AND` operation

![alt text][combined Mag Dir]

In the final step, I combine color filtered image, sobel filtered image and Magnitude & direction filtered image together,

![alt text][Combined_img_or]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_img(img, src, dst)`, which appears in lines 137 through 144 in the file `Adv_lane_detect_clean.py`.  The `warp_img(img, src, dst)` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 200, 0        | 
| 735, 460      | 1080, 0      |
| 250, 675     | 200, 720      |
| 1130, 675      | 1080, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][WarpedImage]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `find_lane(img)` and `fit_use_prev(binary_warped, left_fit_prev, right_fit_prev)` identify lane lines and fit a second order polynomial to both right and left lane lines.
The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position of the left and right lane lines. Here is the histogram of the image,

![alt text][Historgram]

The base points for the left and right lanes are the two peaks nearest the center.

Originally these locations were identified from left and right halves of the histogram, but in my final implementation I changed these to 200 pixels left and right of the midpoint. This helped to reject lines from adjacent lanes. 
The function then identifies ten windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. 
This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image as following image shows,

![alt text][SlideWidows]

Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. 

The `fit_use_prev(binary_warped, left_fit_prev, right_fit_prev)` function is doing the same task, but alleviates much difficulty of the search process by leveraging a previous fit and only searching for lane pixels within a certain range of that fit. 
The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:

![alt text][fit_use_prev]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 303 through 340 in my code in `Adv_lane_detect_clean.py`

The radius of curvature is based upon the following formula:

```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

In this example, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. 
y_0 is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). 
y_meters_per_pixel is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

```
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```
r_fit_x_int and l_fit_x_int are the x-intercepts of the right and left fits, respectively. 
This requires evaluating the fit at the maximum y value because the minimum y value is actually at the top. 
The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 343 through 383 in my code in `Adv_lane_detect_clean.py` in the function `draw_lane(original_img, binary_img, l_fit, r_fit, Minv, curv_rad, center_dist)`.  Here is an example of my result on a test image:

![alt text][draw_lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.

Here's a [link to my video result](./project_video_output.mp4)

Here's a [link to my challenge video result](./challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems during tuning parameters mainly due to the light changing, shadows and the dark spots on the ground.
At first I put very strict thresholds on the image filter. But that filtered too much information. This strategy works fine on the images those have very good conditions.
Then I use a relatively loose thresholds on the video frames. But then another problem happens, too much information left on the image. So I think of some way to filter out the noise.

1. Restrict histogram peaks finding area from 200 instead of 0. Because the adjacent lane will give some noise.
2. Fine lanes base on previous frame result. Since the lane position should not change suddenly.
3. Use the average value to determine the final result. This can make the lanes change in a smooth way and also give the assumption for the frame which has no lanes detected.

I've considered a few possible approaches for making my algorithm more robust. 
1. Active thresholds setting. Add a feedback method so thresholds can adjust it by itself.
2. Since left and right lanes are parallel. It should be useful if left/right lane can adjust itself from each other. For example, if it only recognizes one lane, the other lane can be calculated from it.
3. Designating a confidence level for fits and rejecting new fits that deviate beyond a certain threshold.
4. etc.