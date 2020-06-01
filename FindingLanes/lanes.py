import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #apply gaussian blur in order to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #applying canny function
    canny = cv2.Canny(blur, 50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    #polygons = np.array([[(100,height), (550, height), (288, 160)]])
    polygons = np.array([[(80,330), (600, 330), (318, 224) ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line,right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)

    return line_image

# #loading the image into the program
# image = cv2.imread("test_image2.jpg")
#
# #converting the image to grayscale
# lane_image = np.copy(image)
#
# #identifying lane lines/ region of interest
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 25, np.array([]), minLineLength = 10, maxLineGap = 5)
# #average_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1 )
# cv2.imshow("result",combo_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("project_video.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 50, np.array([]), minLineLength = 25, maxLineGap = 5)
    #average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1 )
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
