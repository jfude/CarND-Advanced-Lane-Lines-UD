import pickle
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import glob
import sys
import collections
from collections import Counter
import time
# matplotlib.use('TKAgg')


#############################################################################
### Line class for maintaining deque of coefficients, 
### state of line base position etc..
###  
#############################################################################
class Line():
    def __init__(self,ypts_in,base_pos_range):

        # uniform set of abscissa ypts 
        self.ypts = ypts_in
        # current set of fit coefficients
        self.current_fit_coeff = np.array([0,0,0], dtype='float')
        # current set of xpts generated from most recent fit 
        self.current_fit_xpts  = np.zeros_like(ypts_in)
        # deque holding coefficients from n most recent
        self.NFC = 14
        self.q_fit_coeff       = collections.deque(self.current_fit_coeff,self.NFC)
        # best fit coefficients, average over n fittings
        self.best_fit_coeff    = self.current_fit_coeff 
        # line base position in pixels
        self.line_base_pos     = 0
        # line base reset 
        self.base_pos_range    = base_pos_range
        #self.base_pos_reset = True
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos_m = None 
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 

        # array of 'anchor points' , points saved from the bottom rectangles
        # to 'anchor' the lane line when there is no data, e.g. when the 
        # lane line is dashed, not solid
        self.anchorx = np.array([],dtype='float')
        self.anchory = np.array([],dtype='float')

        ## The array of fit data based on the best fit coefficients
        self.best_fit_xpts = self.current_fit_xpts 
        

     
    # Reset the entire deque of fit coefficients to a single
    # set of coefficients
    def reset_q_fit(self,coeff):
        # copy current xpts across the deque
        self.current_fit_coeff = coeff
        for i in range(self.NFC):
            self.q_fit_coeff.append(self.current_fit_coeff)



    # Add the latest set of fit coefficients
    def addFitCoeffs(self,coeff):
        # add fit coefficients
        self.current_fit_coeff = coeff
        # update the deque
        self.q_fit_coeff.append(self.current_fit_coeff)
        # update the best fit / do faster update later
        # Don't need to average every time, just maintin list, push and pop
        # for update
        self.best_fit = np.mean(self.q_fit_coeff,axis=0)
        # update xpts
        self.current_fit_xpts = coeff[0]* self.ypts**2 + coeff[1]*self.ypts + coeff[2]
        self.best_fit_xpts    = self.best_fit[0] * self.ypts**2 + self.best_fit[1]*self.ypts + self.best_fit[2]

    def base_pos_reset_needed(self):
        if(self.line_base_pos < self.base_pos_range[0] or self.line_base_pos > self.base_pos_range[1]):
            return True
        

######################################################################################        
        
            
        
        
###################################################################################### 
## camera_calibration:
#         Input: names of calibration files, specified as glob with wildcard
#                e.g. ./calibration_files*.jpg
######################################################################################
def camera_calibration(glob_name):
    
## Read list of calibration image names
    images = glob.glob(glob_name)

    objpoints = [] ## 3d points in space
    imgpoints = [] ## 2d points in image plane

    objp       = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    ret = [False] * len(images)
    n = -1

    for fname in images:

        n += 1
        img = mpimg.imread(fname)
    
        # Convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        ret[n], corners = cv2.findChessboardCorners(gray,(9,6),None)
        
        if ret[n] == True:

            imgpoints.append(corners)
            objpoints.append(objp)
            

    nFalse = Counter(ret)[0]
    nTrue  = Counter(ret)[1]

    print("--Camera_Calibration ",nTrue , "images found out of ",nFalse+nTrue)
    
    # Calibrate if at least one good image
    if(nTrue > 0):
        cal_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, gray.shape[::-1], None,None)
        
    return cal_ret, mtx, dist
######################################################################################        




######################################################################################
### yw_select: Threshold color image for high red and green components
def yw_select(img):
    rg_diff = 30
    rl      = 210
    gl      = 210
    red     = img[:,:,2]
    green   = img[:,:,1]
    diff    = abs(red-green)
    binary_output = np.zeros_like(red)
    binary_output[(diff < rg_diff) & (red > rl) & (green > gl)] = 1 
    return binary_output
######################################################################################
       

######################################################################################
### hls_select : Converts image to hls space, returns binary b/w images where white 
###              statisfies threshold on saturation channel
###   
def hls_select(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    ## Set the threshold here, based on the average saturation in the scene
    s_ave = round(np.mean(s_channel.flatten()))
    thresh = (min(10  + s_ave,255),255)
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
#######################################################################################

 

#################################################################################
##
## abs_sobel_thresh: Takes gray scale image, option 'x' or 'y' threshold limits, returns
###                  b/w binary image where white meets threshold of either 'x' or 'y'
###                  on Sobel detection  
def abs_sobel_thresh(gray, orient='x', kernel=3,thresh=(0,255)):
    # Convert to grayscale                                                                
    # Apply x or y gradient with the OpenCV Sobel() function 
    # and take the absolute value                                                       
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer                                                     
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold                                             
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too             
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result                                                                 
    return binary_output
#################################################################################


#################################################################################
###
### frame_lane_base_pos: Estimate lane base positions based on histogram summing
###                      method 
def frame_lane_base_pos(binary_warped,leftbl,rightbl):
    
    n = round(binary_warped.shape[0]/2)

    histogram = np.sum(binary_warped[n:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint    = np.int(histogram.shape[0]/2)
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print("leftx_base = ",leftx_base)
    #print("rightx_base = ",rightx_base)
    
    ret = True
    if( (leftx_base < leftbl[0]) or (leftx_base > leftbl[1])):
        print("Left failed finding good starting point.")
        ret = False
    if( (rightx_base < rightbl[0]) or (rightx_base > rightbl[1])):
        print("Right failed finding good starting point.")
        ret = False
    
    return ret,leftx_base,rightx_base
##################################################################################



    
    
##################################################################################
###
### Combines the gradient (abs_sobel_thresh) and color (hls_select) threshold,
### returns the AND of these. 
def threshold_frame(img):
    
    
    ## basic img processing
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    ### grad threshold 
    grad_thresh   = abs_sobel_thresh(gray,'x',3,(4,255)) 
    
    ### color threshold
    #color_thresh         = hls_select(img)

    color_thresh         = (hls_select(img) == 1) | (yw_select(img) ==1) 
    ### Make a combined threshold
    combined_thresh = np.zeros_like(grad_thresh)
    combined_thresh[ (grad_thresh == 1) & ( (color_thresh == 1) )] = 1 
    return combined_thresh
###################################################################################    



###################################################################################
###
### frame_len_fit: Determines points within window slices, and then fits a quadratic
###                to these to estimate lane lines. Returns fit coefficients.
###                This was taken from the instructional
###                material, minpix was changed to 150. Also, note mods related to saving
###                'anchoring' points at the base positions. 
def frame_lane_fit(binary_frame,leftLane,rightLane):

        
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_frame.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current  = leftLane.line_base_pos
    rightx_current = rightLane.line_base_pos
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 150
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        # Should set these outside of loop
        win_y_low = binary_frame.shape[0] - (window+1)*window_height
        win_y_high = binary_frame.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        #              (0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        #              (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            #print("left recentering...\n")
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if(window==0):
                addLeftAnchor=False
                leftLane.line_base_pos = leftx_current
                leftLane.anchorx =  nonzerox[good_left_inds] # Save points
                leftLane.anchory =  nonzerox[good_left_inds]
        else:
            if(window==0):
                addLeftAnchor=True

        if len(good_right_inds) > minpix: 
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            if(window==0):
                addRightAnchor=False
                rightLane.line_base_pos = rightx_current
                rightLane.anchorx  = nonzerox[good_right_inds] # Save points
                rightLane.anchory  = nonzeroy[good_right_inds]
        else:
            if(window==0):
                addRightAnchor=True

            
            
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    
    
    if(addLeftAnchor==True):
        leftxx = np.concatenate([leftx,leftLane.anchorx])
        leftyy = np.concatenate([lefty,leftLane.anchory])
    else:
        leftxx = leftx
        leftyy = lefty

    if(addRightAnchor==True):
        rightxx = np.concatenate([rightx,rightLane.anchorx])
        rightyy = np.concatenate([righty,rightLane.anchory])
    else:    
        rightxx = rightx
        rightyy = righty
        

    #out_img[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]]  = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    # Fit a second order polynomial to each
    left_fit = np.polyfit(leftyy, leftxx, 2)
    right_fit = np.polyfit(rightyy, rightxx, 2)
    
    
    return left_fit,right_fit
############################################################################    



############################################################################
###
###  radius_of_curvature: Formula for radius of curvature of lane lines, depending
###                       on the fit coefficients (quadratic fit of lane line) and
###                       the y (vertical axis) base position in the image
###                       
def radius_of_curvature(fit_coeff,yeval):

    rad = ((1 + (2*fit_coeff[0]*yeval + fit_coeff[1])**2 ) **1.5) / np.absolute(2*fit_coeff[0])
    return rad 
############################################################################
    




############################################################################
###
### Main Routine
############################################################################
if __name__ == '__main__':
    

   ### Camera calibration ###
    loadMD = False
    print("Calibrating camera...")
    if(loadMD==False):
        ## Get warp matrix and distortion coefficients
        ret,mtx,dist = camera_calibration("./calibration_images/calibration*.jpg")
        if(ret=="False"):
            print("Error:cameral_calibration failed!") 
            sys.exit(0)
        with open ('cam_cal.pkl','wb') as f:
            pickle.dump([mtx,dist],f)
    else:
        with open('cam_cal.pkl','rb') as f:
            mtx,dist = pickle.load(f)
    # should check load
    ##########################
    
    ###
    ym_per_pix = 30/720.0  # meters per pixel in y dimension
    xm_per_pix = 3.7/700.0 # meters per pixel in x dimension
    ###    

    ### Parameters for writing text to frames
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    upperLeftCornerOfText = (10,100)
    fontScale              = 0.5
    fontColor              = (0,255,255)
    lineType               = 1



    
    
    ### Load video
    video = cv2.VideoCapture('./project_video.mp4')
    rval = video.isOpened()
    if(rval == False):
        print("Video failed to open.")
        sys.exit(0)
        
    
    ### Sanity check on base  locations
    leftbl = (100,500)
    rightbl= (850,1150)    

    
    ### Source positions for perspective transform
    tl = (594, 451)
    bl = (270,  670)
    br = (1032, 670)
    tr = (685 ,451)
    src = np.float32([[tl],[bl],[br],[tr]])
    
    ### Destination points for perspective transform
    tl = (370,0)
    bl = (370,720)
    br = (960,720)
    tr = (960,0)
    dst = np.float32([[tl],[bl],[br],[tr]])  

    ### Forward and inverse perspective transforms
    M  = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    

    ### Fit coefficients in real space (meters)
    left_fit_m = np.zeros((3),np.float32)
    right_fit_m = np.zeros((3),np.float32)
    
    
    ## Read first frame to get frame size
    rval, frame = video.read()
    ypts = np.linspace(0, frame.shape[0]-1, frame.shape[0] )
    
    ## Image frame center (in meters). Zero (0) is at the left of the image
    xframe_center =  (frame.shape[1]/2.0) * xm_per_pix  # meters per pixel in x dimension
    
    ## Create lane lines 
    ##     ypts: Uniform set of abscissa points on which the lane lines are evaluated
    ##     The second arguments are boundaries outside of which a reset of the base position
    ##     is called for.
    leftLane  = Line(ypts,(300,550))
    rightLane = Line(ypts,(920,1150))

    
    
    frameStart = 0
    for i in range(frameStart):
        rval, frame = video.read()
        if(rval==False):
            print("Failed to read frame.")
            sys.exit(0)
        
    

    fCount = frameStart 
    initAve = True

    ### Sorensen video 3 codec is used for video output
    fourcc = cv2.VideoWriter_fourcc('S', 'V', 'Q', '3')
    video_writer = cv2.VideoWriter('./project_output.sv3',fourcc,20.0,(1280,720),True)
    if(video_writer.isOpened() == False):
        print("video_writer failed to open")
        sys.exit(0)
    

    ### Loop through image frames
    while(True):
        fCount +=1
        #print("Reading frame #",fCount)
        rval, frame = video.read()
        if(rval==False):
            print("Failed to read frame.")
            break
    
        ### Undistort according to camera calibration mtx,dist
        uframe  =  cv2.undistort(frame,mtx,dist,None,mtx)
        ### Warp perspective to overhead view
        uwframe =  cv2.warpPerspective(uframe, M, (frame.shape[1],frame.shape[0]), flags=cv2.INTER_LINEAR)
        
        ### Apply Gradient/Color threshold
        binary_frame = threshold_frame(uwframe)
        


        #binary_frame = threshold_frame(uwframe)
        ## Might want to do each individually
        if( leftLane.base_pos_reset_needed() or rightLane.base_pos_reset_needed()):
            # print("!!Base pos reset...")
            rval,leftx_base,rightx_base = frame_lane_base_pos(binary_frame,leftbl,rightbl)
            # Set base points
            leftLane.line_base_pos    = leftx_base
            rightLane.line_base_pos   = rightx_base
                
            if(rval==False):
                print(" failed.")
                continue
            else:
                ## suceeded 
                leftLane.base_pos_reset   = False
                rightLane.base_pos_reset  = False
        
        
        #out_img = np.dstack((binary_frame, binary_frame, binary_frame))*255
        left_fit,right_fit = frame_lane_fit(binary_frame,leftLane,rightLane)
        
        
        
        
        ## Plot lane line fit with current best fit
        
        if(initAve):
            leftLane.reset_q_fit(left_fit)
            rightLane.reset_q_fit(right_fit)
            leftLane.addFitCoeffs(left_fit)
            rightLane.addFitCoeffs(right_fit)
            initAve=False

        else:
            left_xpts  = left_fit[0]*ypts**2 + left_fit[1]*ypts + left_fit[2]
            right_xpts = right_fit[0]*ypts**2 + right_fit[1]*ypts + right_fit[2]
            diff_xpts  = right_xpts - left_xpts
            #print("max_diff_xpts = ", max(diff_xpts), " min_diff_xpts = ",min(diff_xpts)) 
            if(max(diff_xpts) < 750 and min(diff_xpts) > 530):
                leftLane.addFitCoeffs(left_fit)
                rightLane.addFitCoeffs(right_fit)

    
        # Recast the x and y points into usable format for cv2.fillPoly()
        
        pts_left  = np.array([np.transpose(np.vstack([leftLane.best_fit_xpts, leftLane.ypts]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLane.best_fit_xpts, rightLane.ypts])))])
        pts = np.hstack((pts_left, pts_right))

        warp_zero  = np.zeros_like(binary_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0])) 
        # Combine the result with the original image
        result =  cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        
        

        ## Compute the fit coefficients in real space and then the radius of curvature
        left_fit_m[2] = xm_per_pix                 * left_fit[2]
        left_fit_m[1] = (xm_per_pix/ym_per_pix)    * left_fit[1]
        left_fit_m[0] = (xm_per_pix/ym_per_pix**2) * left_fit[0]
        
        right_fit_m[2] = xm_per_pix                 * right_fit[2]
        right_fit_m[1] = (xm_per_pix/ym_per_pix)    * right_fit[1]
        right_fit_m[0] = (xm_per_pix/ym_per_pix**2) * right_fit[0]
        
        yeval = frame.shape[0]*ym_per_pix
        left_rc = radius_of_curvature(left_fit_m,yeval)
        right_rc = radius_of_curvature(right_fit_m,yeval)

        ### Compute offset = difference between frame center and lane center
        offset =  xframe_center - \
        (rightLane.best_fit_xpts[-1] + leftLane.best_fit_xpts[-1])*xm_per_pix/2.0  


        fstr = "Frame# {0}".format(fCount) +  "   Left(m):{0:.3f}".format (left_rc) + "  Right(m):{0:.3f}".format(right_rc) +  " Offset(m):{0:.3f}".format(offset) 
        
        print('\r',fstr,end='')        

        cv2.putText(result,
                    fstr,
                    upperLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        #cv2.imshow("name",result)
        video_writer.write(result)

        
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()    

print("\n Closed Video.\nDone.")
    



