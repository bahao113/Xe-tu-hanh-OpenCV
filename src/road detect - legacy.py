import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):

    heading_image = np.zeros_like(frame)
    height = frame.shape[0]
    width = frame.shape[1]

    steering_angle_radian = steering_angle / 180.0 * np.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    heading_image = cv.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def get_steering_angle(frame, lane_lines):
    height = frame.shape[0]
    width = frame.shape[1]

    if len(lane_lines) == 2:  # if two lane lines are detected
        _, _, left_x2, _ = lane_lines[0][0]  # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0]  # extract right x2 from lane_lines array
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:  # if only one line is detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:  # if no line is detected
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10
    line_segments = cv.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=0)
    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6): # line color (B,G,R)
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def make_points(frame, line):
    height = frame.shape[0]
    width = frame.shape[1]
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segment detected")
        return lane_lines

    height = frame.shape[0]
    width = frame.shape[1]
    left_fit = []
    right_fit = []
    boundary = 1/3

    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity)")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # lane_lines is a 2-D array consisting the coordinates of the right and left lane lines
    # for example: lane_lines = [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    # where the left array is for left lane and the right array is for right lane
    # all coordinate points are in pixels
    return lane_lines

def region_of_interest(img, vertices):
    mask_func = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv.fillPoly(mask_func, vertices, match_mask_color)
    # return
    masked_image = cv.bitwise_and(img, mask_func)
    return masked_image


def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)


    else:
        return img


    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    # cv.imshow('bbb', blank_image)
    # cv.waitKey(0)
    return img

cap = cv.VideoCapture('line3.mp4')
while True:
    ret, frame = cap.read()
    if ret is not True:
        break
    image = frame
    gray = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # cv.imshow('aaa', gray)
    # cv.waitKey(0)
    lower_blue = np.array([90, 120, 0], dtype = "uint8") # lower limit of blue color
    upper_blue = np.array([150, 255, 255], dtype="uint8") # upper limit of blue color
    mask = cv.inRange(gray, lower_blue,upper_blue) # this mask will filter out everything but blue
    # cv.imshow('aaa', mask)
    # cv.waitKey(0)
    height = image.shape[0]
    width = image.shape[1]

    polygon = np.array([[
            (0, height),
            (0,  height/2),
            (width , height/2),
            (width , height),
        ]], np.int32)

    mask = region_of_interest(mask, polygon)
    edges = cv.Canny(mask, 50, 100)
    # cv.imshow('aaa', mask)
    # cv.waitKey(0)

    lines_seg = detect_line_segments(edges)
    lane_lines = average_slope_intercept(edges, lines_seg)

    line_image = display_lines(image,lane_lines)
    # cv.imshow('aaa', line_image)
    # cv.waitKey(0)
    angle = get_steering_angle(edges, lane_lines)
    heading = display_heading_line(line_image, angle)
    cv.imshow('aaa', heading)
    if cv.waitKey(25) == ord('q'):
        break
# img = cv.imread('auto.png')

# cv.imshow('bbb', cropped_image)
# cv.waitKey(0)
# lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 30, threshold=50, lines=np.array([]), minLineLength=150,
#                        maxLineGap=100)
# image_with_line = drow_the_lines(mask, lines)
# cv.imshow('aa', image_with_line)

cv.waitKey(0)
cv.destroyAllWindows()