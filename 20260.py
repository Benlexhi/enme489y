import cv2
import numpy as np

camera = cv2.imread("testimage_Spring2026_yellow.jpg")
cv2.imshow("Original frame", camera)

camera_copy = cv2.imread("testimage_Spring2026_yellow.jpg")
box = cv2.rectangle(camera_copy, (0, 260), (640, 400), (0, 255, 0), 0)
cv2.imshow("Rectangle", box)

#define points array for transform
pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
pts2 = np.float32([[0, 0], [480, 0], [0, 640], [480, 640]])

#Apply perspective transform alg.
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(camera_copy, matrix, (640, 480))

#Transformed capture
cv2.imshow("frame1", result)

cv2.waitKey(0)