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

[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

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


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After calibrating the camera I tested the unidstortion on the test images:

![alt text][image3]


#### 2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

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


#### 3. Describe how you performed a perspective transform and provide an example of a transformed image.

My transformation of perspective was hard-coded the following way:
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


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
