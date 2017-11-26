

## Advanced Lane Finding Project 

The goal of this project is to demonstrate a beginning method for 
identifying lane lines for a self-driving car, appropriate for a fairly flat road. 
The processing steps that are covered in this project are the following:


* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[original_cal2_img]:./calibration_images/calibration2.jpg  
[corrected_cal2_img]:./project_images/corrected_calibration2.jpg  
[original_sample_camera_img]:./project_images/signs_vehicles_xygrad.jpg  
[corrected_sample_camera_img]:./project_images/corrected_signs_vehicles_xygrad.jpg  
[orig_undist_perspective_img]:./project_images/orig_undist_perspective.png  
[overhead_perspective_img]:./project_images/overhead_persective.png  
[overhead_after_threshold_img]:./project_images/overhead_after_threshold.png  
[pixel_to_real_img]:./project_images/pixel_to_real.png


## Rubric Points
Here I will provide a reference to the sections below that address each individual rubric. The rubric points and descriptions for this project may be found [here](https://review.udacity.com/#!/rubrics/571/view).

- [Camera Calibration](#camera-calibration)
- Pipeline and Test Images
  - [Distortion Corrected Image](#distortion-corrected-image)
  - [Color and Gradient Threshold](#color-and-gradient-threshold)
  - [Perspective Transform](#perspective-transform)
  - [Lane Line Finding](#lane-line-finding)
  - [Lane Curvature and Offset](#lane-curvature-and-offset)
  - [Example Lane Identification Image](#example-lane-identification-image)  

- [Pipeline Video](#pipeline-video)
- [Discussion](#discussion)
 

## Camera Calibration

Processing of images for lane identification begins with calibration of the camera
using the set of chess board images found in CarND-Advanced-Lane-Lines/camera_cal.
The code for the camera calibration in the main program lane_finding.py can be found in the
function camera_calibration(glob_name). We seek to calculate the transformation matrix and
distortion coefficients that represents how the mapping of pixel space to real space changes
across a camera image. These are the properties of the camera itself and are calculated 
by finding the b/w square intersections, or corners, in the image and mapping to the x,y 
plane (z=0).  

For each read calibration image, we first convert to gray scale and the findChessboardCorners
cv2 function, with the expectation that there are 9x6 interior corners to be found. If these 
corners are found they are added to a list. A parallel list of the corners in the z=0 plane
is also added to and maintained. It was found that the 9x6 corner set was detected in  
17 out of the 20 images; in the 3 of the images some outer corners fell outside of the image.

The function cv2.calibrateCamera is called with these lists and returns the estimated transformation matrix mtx and distortion coefficients dist. The distortion correction is the applied via the cv2 function undistort.

An example of the distortion correction applied to the calibration image calibration2.jpg
is shown below.

Original Cal Image
![Original Cal Image][original_cal2_img]

Corrected Cal Image
![Corrected Cal Image][corrected_cal2_img]


## Pipeline and Test Images 

Here we describe each step in the process of identifying lane lines in a camera image or 
video frame and give an example of each. Note that we performed the perspective transform before
thresholding.


### Distortion Corrected Image

The first step in the pipeline is to undistort the image using the transformation matrix and 
distortion coefficients calculated from the calibration images, as described above. Here is an 
example of the application of the distortion correction on a sample camera image.

Original Camera Image
![Original Camera Image][original_sample_camera_img]  
Undistorted Camera Image
![Corrected Camera Image][corrected_sample_camera_img]  


### Perspective Transform

The lane line shape and curvature is much easier to detect when the image is projected into
an overhead view. The transform M is pre-calculated before the image frame loop using 
```
M  = cv2.getPerspectiveTransform(src, dst)
```
and then applied directly after the distortion correction using

```
uwframe =  cv2.warpPerspective(image, M, (frame.shape[1],frame.shape[0]), flags=cv2.INTER_LINEAR) 
```

in the loop. The transform M maps the source points (src) from an original image to 
the destination points (dst), which are chosen to create this birds eye view.


To focus on the lane, src is chosen as an array of four points which roughly represent the lane 
end points, determined by hand using a sample image and hardcoded. The dst points are chosen
to form a front facing (screen or image plane) rectangle. The points are 
    

| Source        | Destination   |
|:-------------:|:-------------:|
| 594, 451      | 370, 0        |
| 270, 670      | 370, 720      |
| 1032, 670     | 960, 720      |
| 685, 451      | 960, 0        |


Here is an example of the application of the perspective transform

Original Undistorted Image
![Original Undistorted Image][orig_undist_perspective_img]   
Overhead Perspective Image
![Overhead Perspective][overhead_perspective_img]



### Color and Gradient Threshold

The third step in our pipeline is the application of a color and gradient threshold, the code for
which can be found in the routine threshold_frame().
For this project, although both thresholds are necessary, it was found that the color threshold
was the most significant. We used a color saturation threshold (converting from RGB to HLS space) 
as was suggested in the teaching material, but we found that **adjusting the bottom threshold  of 
the saturation as basically an offset above the average saturation in the current scene greatly 
improved our results**. Specifically, for each scene the minimim saturation threshold was set
to 10 above the average 

```python
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
s_channel = hls[:,:,2]
## Set the threshold here, based on the average saturation in the scene 
s_ave = round(np.mean(s_channel.flatten()))
thresh = (min(10  + s_ave,255),255)
```


For the gradient threshold we used only a threshold for a gradient in the x or vertical direction
(in image space). The lower bound threshold was quite low, only 4. The transformed color image 
used for the above color thresholding is first converted to gray scale. Then the cv2 Sobel detection
routine is used as follows

```python
abs_sobel = np.absolute(cv2.Sobel(gray)img, cv2.CV_64F, 1, 0))
```

The result is normalized and rescaled back to the 0 to 255 range. The routine used for the gradient thresholding is abs_sobel_thresh().

Below we show an undistorted perspective warped sample image before and after
this thresholding is applied.      
 
 Overhead Perspective Before Threshold
![Overhead Perspective Before Threshold][overhead_perspective_img]
 Overhead Perspective After Threshold
![Overhead Perspective After  Threshold][overhead_after_threshold_img]  



   
### Lane Line Finding 

In the next pipeline step, we attempt to fit a smooth quadratic curve
to the lane lines found in the thresholded image. In our pipeline, we use the histogram 
method only for the first initial estimate of the lane line positions at the bottom of image.

#### Initial lane position estimate

Each pixel column of the bottom half of the threshold image is summed, and the peaks 
in the resulting array are searched for as estimates for these lane "anchor" positions.
 
This procedure is found in the routine frame_lane_base_pos() and is used as a first initial 
estimate. From there we use the sliding window method for finding points to fit to estimate 
a lane line. This method is described in the Advanced Lane Finding:Sliding Window section of the
instructional material. 

#### Sliding window method

A very important modification we made to this routine is to save points found in the bottom rectangles
(anchor points) to use for fitting when in later frames there are none. When more than minpix
pixels are found in the bottom rectangle (the same condition for recentering), the previous  
set of anchor points are replaced with the newly found points. 


#### Smoothing Fit Coefficients

As a way of preventing rapid drastic changes in the lane line estimates,
for example if the thresholded image does not yield clear lane lines and hence a 
bad fit, we use a moving average of the fit coefficients. This is implemented with a 
double ended queue of length twenty.

```python
class Line():
      self.NFC               =      20
      self.q_fit_coeff       =      collections.deque(self.current_fit_coeff,self.NFC)   
````
 
#### Lane Base Position Range Check

In addition, a range check is also applied to the lane base positions. If either the left or right base positions 
are found to be out of bounds, then we return to the histogram method to recalculate the base positions. The new base positions reset the window positions at the bottom of the frame images (the base position) in the sliding window method.


### Lane Curvature and Offset

In order to calculate the lane radii of curvature, it is necessary to determine the previously discussed fit coefficients in real space. 

The conversion from pixel space to real space for these coefficients is straightforward.

Pixel space to Real Space Conversion
![Pixel space to Real Space Conversion][pixel_to_real_img]  

The radius of curvature is calculated in the following routine in lane_finding.py

```python
def radius_of_curvature(fit_coeff,yeval):

    rad = ((1 + (2*fit_coeff[0]*yeval + fit_coeff[1])**2 ) **1.5) / np.absolute(2*fit_coeff[0])
    return rad
```

For each video frame, the radius of curvature for the left and right lane as well as an offset 
is calculated. The offset is given by the difference of the midpoint of the lane line estimate
and the center of the given frame along the x-direction. In the code this expressed as

```python 
offset =  xframe_center - \
        (rightLane.best_fit_xpts[-1] - leftLane.best_fit_xpts[-1])*xm_per_pix/2.0
```

where xm_per_pix (see lane_finding.py) converts quantities from pixel space to meters. xframe_center is already expressed 
in meters.
 

### Example Lane Identification Image

Below is an example frame image from the video, where the lane overlay is plotted back down on the original image.



The output of lane_finding.py corresponding to this frame is 
```
Frame#  1   Left(m):  462.175621668    Right(m):  498.90262862    Offset(m):  -0.434047384007
```
where left and right here refer to the radii of curvature.


## Pipeline Video

The lane_finding.py program reads the video project_video.mp4 and 
plots the lane estimation on each frame. The lane estimation for the project video
is pretty stable with slight distortion over the sections of the road where there is a paved
road/concrete section transition. This typically corresponds to transitions from low to high (or high/low) saturation image frames. The output of the program is project_output.sv3. My output is saved in project_output_final.sv3 and project_output_final.mov.

## Discussion



  








