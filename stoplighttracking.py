# ENME 489Y: Remote Sensing
#Check for updated version

# Python script tracks green 'stoplight'
# and saves video of tracking to stoplight.mp4

# import the necessary packages
from picamera.array import PiRGBArray
from picamera2 import Picamera2
import numpy as np
import imutils
import cv2
import time

# define the lower and upper boundaries of the
# green circle in the HSV color space
# Note: use colorpicker.py to create a new HSV mask
colorLower = (29, 70, 6)
colorUpper = (75, 255, 255)

# initialize the Raspberry Pi camera
picam2 = Picamera2()

config = picam2.create_video_configuration(
	main={"size": (640, 480)},
	controls={"Framework":  25}
)

picam2.configure(config)

picam2.start()
time.sleep(0.1)

#camera.resolution = (640, 480)
#camera.framerate = 25
#rawCapture = PiRGBArray(camera, size=(640,480))

# allow the camera to warmup
time.sleep(0.1)

# define the codec and create VideoWriter object
# UNCOMMENT THE FOLLOWING TWO (2) LINES TO SAVE .avi VIDEO FILE
# TRY BOTH XVID THEN MJPG, IN THE EVENT THE .avi FILE IS NOT SAVING PROPERLY
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('stoplight.avi',fourcc,10,(640, 480))


# keep looping
try:
	while True:
		# Grab a frame (picamera2 captures on demand)
		frame = picam2.capture_array()

		# Convert from RGB (picamera2 default) to BGR for OpenCV
		image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		# blur the frame and convert to the HSV color space
		blurred = cv2.GaussianBlur(image, (11, 11), 0)
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# construct a mask for the color "green"
		mask = cv2.inRange(hsv, colorLower, colorUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			if radius > 0:
				cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.circle(image, center, 2, (0, 0, 255), -1)
			# out.write(image)  # Uncomment to save video

		# show the frame
		cv2.imshow("Frame", image)
		key = cv2.waitKey(1) & 0xFF

		# press 'q' to quit
		if key == ord("q"):
			break

except KeyboardInterrupt:
	pass

# Cleanup
cv2.destroyAllWindows()
picam2.stop()