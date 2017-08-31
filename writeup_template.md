## Writeup
## Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

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

[image1]: ./writeup_imgs/original_undist.png "Undistorted"
[image2]: ./writeup_imgs/chessboard.png "Chessboard Corners"
[image3]: ./writeup_imgs/test_udist.png "Image Unidstorted"
[image4]: ./writeup_imgs/combined_binary.png "Combined binary thresholds"
[image5]: ./writeup_imgs/undist_warped.png "Unidstored and warped"
[image6]: ./writeup_imgs/image_pipeline.png "Pipeline image"

[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup


### Camera Calibration

#### 1. Camera matrix and distortion coefficients.

The first step was to find the chessboard corners:

````
nx = 9 
ny = 6 
images = glob.glob("camera_cal/calibration*.jpg")
objpoints = []
imgpoints = []
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for fname in images:
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    if ret == True:
        
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
       
````

![Chessboard Corners][image2]

As a second step for my solution, I computed the camera matrix and distortion coefficients and then saved them to a pickle so I can read them on the image pipeline:

````
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg',dst)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_calibration_result.p", "wb" ) )
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted', fontsize=15)
````

![Original Vs Unidstorted][image1]


### Pipeline (images)

#### 1. Distortion-corrected image.

After calibrating the camera I tested the unidstortion on the test images:

![alt text][image3]


#### 2. Color transforms and gradients to create a thresholded binary image. 

The process to obtain a binary image was:
* Convert the image to grayscale
* Take the absolute sobel derivative of the image with respect to x
* Normalize the sobel derivative
* Threshold the gradient
* Convert the image to HLS c-space and separate the S channel
* Threshold the S channel
* Combine the two binaries

````
def app_thresh(image, xgrad_thresh=(20,100), s_thresh=(170,255)):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))    
    n_sobel = np.uint8(255*sobelx/np.max(sobelx))    
    sxbinary = np.zeros_like(n_sobel)
    sxbinary[(n_sobel >= xgrad_thresh[0]) & (n_sobel <= xgrad_thresh[1])] = 1
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary
````

The result was an image like this:

![alt text][image4]


Sometimes the right lane was not identified, I tried by adjusting a bit the thresholds with a while loop:

````
xgrad_thresh_temp = (40,100)
s_thresh_temp=(125,255)

    while fits == False:
        combined_binary = app_thresh(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)
        warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        leftx, lefty, rightx, righty = hist_pixels(warped, horizontal_offset=40)
        if len(leftx) > 1 and len(rightx) > 1:
            fits = True
        xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
        s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)

````

#### 3. Perspective transform.

My transformation of perspective was hard-coded by trial and error and it's final form was:
````
h,w = raw.shape[:2]

src = np.float32([(575,464),
(707,464),
(258,682),
(1049,682)])

dst = np.float32([(280,0),
(w-250,0),
(280,h),
(w-250,h)])
````

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 280, 0        | 
| 707, 464      | 1030, 0       |
| 258, 682      | 280, 720      |
| 1049, 682     | 1030, 720     |


The transformation on the image was:

![alt text][image5]


#### 4. Identify lane-line pixels and fit their positions with a polynomial

By combining the step 2 and this step (4) I was able to fit a polynomial most of the frames of the video and was able to reduce the flickering by evaluating if the detected lanes were similar to the previous lanes. If they were not similar, I re used the previous lanes. (see discussion section).


````
def hist_pixels(warped_thresholded_image, offset=50, steps=6,
window_radius=200, medianfilt_kernel_size=51,
horizontal_offset=50):

    
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    height = warped_thresholded_image.shape[0]
    offset_height = height - offset
    width = warped_thresholded_image.shape[1]
    half_frame = warped_thresholded_image.shape[1] // 2
    pixels_per_step = offset_height / steps

    for step in range(steps):
        left_x_window_centres = []
        right_x_window_centres = []
        y_window_centres = []

        window_start_y = height - (step * pixels_per_step) + offset
        window_end_y = window_start_y - pixels_per_step + offset

        
        histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y), int(horizontal_offset):int(width - horizontal_offset)], axis=0)

        
        histogram_smooth = signal.medfilt(histogram, medianfilt_kernel_size)

        left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
        right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
        if len(left_peaks) > 0:
            left_peak = max(left_peaks)
            left_x_window_centres.append(left_peak)

        if len(right_peaks) > 0:
            right_peak = max(right_peaks) + half_frame
            right_x_window_centres.append(right_peak)

        if len(left_peaks) > 0 or len(right_peaks) > 0:
            y_window_centres.append((window_start_y + window_end_y) // 2)

        # pixels left window
        for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
            left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre,
                                                                       y_centre, window_radius)

            left_x.append(left_x_additional)
            left_y.append(left_y_additional)
            
        for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
            right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre,
                                                                         y_centre, window_radius)
            right_x.append(right_x_additional)
            right_y.append(right_y_additional)

    if len(right_x) == 0 or len(left_x) == 0:
        print("Init no peaks for left or right")
        print("left_x: ", left_x)
        print("right_x: ", right_x)

        horizontal_offset = 0

        left_x = []
        left_y = []
        right_x = []
        right_y = []

        for step in range(steps):
            left_x_window_centres = []
            right_x_window_centres = []
            y_window_centres = []

            
            window_start_y = height - (step * pixels_per_step) + offset
            window_end_y = window_start_y - pixels_per_step + offset

            
            histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y),
                               int(horizontal_offset):int(width - horizontal_offset)], axis=0)

            histogram_smooth = signal.medfilt(histogram, medianfilt_kernel_size)

            
            left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
            right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
            if len(left_peaks) > 0:
                left_peak = max(left_peaks)
                left_x_window_centres.append(left_peak)

            if len(right_peaks) > 0:
                right_peak = max(right_peaks) + half_frame
                right_x_window_centres.append(right_peak)

            if len(left_peaks) > 0 or len(right_peaks) > 0:
                y_window_centres.append((window_start_y + window_end_y) // 2)

            for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
                left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre,
                                                                           y_centre, window_radius)

                left_x.append(left_x_additional)
                left_y.append(left_y_additional)

            for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
                right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre,
                                                                             y_centre, window_radius)

                right_x.append(right_x_additional)
                right_y.append(right_y_additional)

    return collapse_into_single_arrays(left_x, left_y, right_x, right_y)
    
````


#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature:
````
y_ev = 500
left_curve_radius = np.absolute(((1 + (2 * left_coeffs[0] * y_ev + left_coeffs[1])**2) ** 1.5) \
/(2 * left_coeffs[0]))
right_curve_radius = np.absolute(((1 + (2 * right_coeffs[0] * y_ev + right_coeffs[1]) ** 2) ** 1.5) \
/(2 * right_coeffs[0]))

curvature = (left_curve_radius + right_curve_radius) / 2
````

#### 6. Example image of my result.
The result was a image with the lane area identified:

![alt text][image6]

---

### Pipeline (video)

#### 1. Final video output.

Here's a [link to my video result](./project_video.mp4)



---

### Discussion


* The first problem I faced was extreme wobbling, in particular with the frames that had tree shadows on the pavement. To make the detection more robust I used a function to evaluate the found lanes. 

````
def continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs, prev_right_coeffs):
   if prev_left_coeffs == None or prev_right_coeffs == None:
      return True
   b_left = np.absolute(prev_left_coeffs[1] - left_coeffs[1])
   b_right = np.absolute(prev_right_coeffs[1] - right_coeffs[1])
   if b_left > 0.5 or b_right > 0.5:
      return False
    else:
      return True

````
on the image pipeline I used the past coefficients to plot the detected lanes if the lanes were too different (lanes dont change much from frame to frame).

````
if not continuation_of_traces(left_coeffs, right_coeffs, past_left_coefs, past_right_coefs):
   if past_left_coefs is not None and past_right_coefs is not None:
      left_coeffs = past_left_coefs
      right_coeffs = past_right_coefs
         
````
* Another problem was too much noise in the binary image generated by the thresholds on the X-gradient. To fix this I set the thresholds to (40, 100).

* Finally, to make the detection more robust more image filters can be applied.

