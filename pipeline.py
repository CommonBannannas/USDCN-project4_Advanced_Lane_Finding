"""
image processing pipeline
"""
from utils import *

#open camera calibration results:
with open("camera_calibration_result.p", mode='rb') as f:
    camera_calib = pickle.load(f)
mtx = camera_calib["mtx"]
dist = camera_calib["dist"]

past_left_coefs = None
past_right_coefs = None

def image_pipeline(file, filepath=False):
    global past_left_coefs
    global past_right_coefs

    plt.clf()

    if filepath == True:
        # Read in image
        raw = cv2.imread(file)
    else:
        raw = file

    # Parameters
    imshape = raw.shape
    h,w = raw.shape[:2]

    src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])

    dst = np.float32([(280,0),
                  (w-250,0),
                  (280,h),
                  (w-250,h)])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    height = raw.shape[0]
    offset = 50
    offset_height = height - offset
    half_frame = raw.shape[1] // 2
    steps = 12
    pixels_per_step = offset_height / steps
    window_radius = 200
    medianfilt_kernel_size = 51

    # set a blak canvas
    b_canvas = np.zeros((720, 1280))
    col_canvas = cv2.cvtColor(b_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Apply distortion correction to raw image
    image = cv2.undistort(raw, mtx, dist, None, mtx)
    combined = app_thresh(image)

    fits = False
    #curv_check = False

    xgrad_thresh_temp = (40,100)
    s_thresh_temp=(150,255)

    while fits == False:
        combined_binary = app_thresh(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)
        # Warp onto birds-eye-view
        # Previous region-of-interest mask's function is absorbed by the warp
        warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        # Histogram and get pixels in window
        leftx, lefty, rightx, righty = hist_pixels(warped, horizontal_offset=40)
        if len(leftx) > 1 and len(rightx) > 1:
            fits = True
        xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
        s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)

    # curvature of the lane
    # y-value where we want radius of curvature
    y_ev = 500
    left_curve_radius = np.absolute(((1 + (2 * left_coeffs[0] * y_ev + left_coeffs[1])**2) ** 1.5) \
                    /(2 * left_coeffs[0]))
    right_curve_radius = np.absolute(((1 + (2 * right_coeffs[0] * y_ev + right_coeffs[1]) ** 2) ** 1.5) \
                     /(2 * right_coeffs[0]))

    curvature = (left_curve_radius + right_curve_radius) / 2
    min_curve_radius = min(left_curve_radius, right_curve_radius)

    if not continuation_of_traces(left_coeffs, right_coeffs, past_left_coefs, past_right_coefs):
            if past_left_coefs is not None and past_right_coefs is not None:
                left_coeffs = past_left_coefs
                right_coeffs = past_right_coefs


    past_left_coefs = left_coeffs
    past_right_coefs = right_coeffs

    # determine vehicle position with respect to center
    centre = center(719, left_coeffs, right_coeffs)

    # warp the detected lane boundaries back onto the original image.
    polyfit_left = draw_poly(b_canvas, lane_poly, left_coeffs, 40)
    polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 40)

    # convert to color and highlight lane line area
    trace = col_canvas
    trace[polyfit_drawn > 1] = [150,0,255]

    area = highlight_lane_line_area(b_canvas, left_coeffs, right_coeffs)
    trace[area == 1] = [0,75,255]
    lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    combined_img = cv2.add(lane_lines, image)
    add_figures_to_image(combined_img, curvature=curvature,
                         vehicle_position=centre,
                         min_curvature=min_curve_radius,
                         left_coeffs=left_coeffs,
                         right_coeffs=right_coeffs)
    plt.imshow(combined_img)
    return combined_img

combined_img = image_pipeline("test_images/test1.jpg", filepath=True)
