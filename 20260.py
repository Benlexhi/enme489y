import cv2
import numpy as np

camera = cv2.imread("testimage.jpg")
cv2.imshow("Original frame", camera)

camera_copy = cv2.imread("testimage.jpg")
box = cv2.rectangle(camera_copy, (0, 260), (640, 400), (0, 255, 0), 0)
cv2.imshow("Rectangle", box)

#define points array for transform
pts1 = np.float32([[10, 20], [50, 20], [10, 20]])
pts2 = np.float32([[10, 20], [50, 20], [10, 20]])

#Apply perspective transform alg.
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(camera_copy, matrix, (640, 480))

#Transformed capture
cv2.imshow("frame1", result)

cv2.waitKey(0)