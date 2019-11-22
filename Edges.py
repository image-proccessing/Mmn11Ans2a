# Edges.py
# Written by: Amir Fried
#
# Description:
# This program finds the edges of all adjacent pixels
#
# Requires:
# opencv, numpy, argparse
#
# How to use:
# python Edges.py -f <image file path> -t <threshold>
# Example:
# python Edges.py -f cameraman.jpg -t 100

import argparse
import cv2
import numpy as np

# Arguments management:
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="Path to the image file")
ap.add_argument("-t", "--threshold", required=True, help="Threshold value between 0 and 255")
args = vars(ap.parse_args())
image_file = args["file"]
threshold = float(args["threshold"])

# Image load (in graycycle)
orig_image = cv2.imread(image_file, 0)
height, width = orig_image.shape

# Image conversion to binary per input threshold (inversion is used since contours are found for white)
_, bw_image = cv2.threshold(orig_image, threshold, 255, cv2.THRESH_BINARY_INV)

# finding edges (contours) with CHAIN_APPROX_NONE so it will not remove edge pixels
contours, _ = cv2.findContours(bw_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Displaying edges
empty_image = np.uint8(np.zeros((height, width, 3)))  # creating an all zero color image with same size as the original
cv2.drawContours(empty_image, contours, -1, (255, 255, 255), 1)  # draw the edges with a single pixel white line
cv2.imshow("Edges", empty_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
