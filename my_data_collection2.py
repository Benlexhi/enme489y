# ENME489Y: Remote Sensing

# import the necessary packages
import numpy as np
import time
import cv2
import imutils
import os
from picamera2 import Picamera2

# allow the camera to setup
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={"FrameRate": 25}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# Enter the initial IMU angle from user
d = input("Please enter IMU angle: ")
print("Confirming the IMU angle you entered is: ")
print(d)

# Capture image directly using Picamera2 (replaces raspistill)
image = picam2.capture_array()
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Picamera2 outputs RGB, OpenCV expects BGR
image = cv2.flip(image, -1)

# plot crosshairs for alignment
cv2.line(image, (640, 0), (640, 720), (0, 150, 150), 1)
cv2.line(image, (600, 360), (1280, 360), (0, 150, 150), 1)

# display IMU angle, for reference
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
red = (0, 0, 255)
cv2.putText(image, d, (800, 200), font, 10, red, 10)

# write image to file
d = int(d)
filename = "%d.jpg" % d
cv2.imwrite(filename, image)

os.makedirs('/home/bserrano/ENME435/test_scans', exist_ok=True)
os.rename(filename, '/home/bserrano/ENME435/test_scans' + filename)  # replaces os.system mv

print("All done!")