# ENME489Y: Remote Sensing

# import the necessary packages
from picamera2 import Picamera2
import numpy as np
import time
import cv2
import imutils

# initialize the camera
picam2 = Picamera2()

# Configure camera resolution and framerate
config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={"FrameRate": 25}
)
picam2.configure(config)

# Start the camera
picam2.start()

# allow the camera to warm up
time.sleep(1)

# Main video loop - Picamera2 doesn't have capture_continuous,
# so we use a while loop and manually capture each frame
try:
    while True:
        # Grab a frame from the camera
        # capture_array() returns RGB by default, but OpenCV uses BGR
        # You can also use capture_array("bgr") to get BGR directly
        frame = picam2.capture_array("main")

        # Make a copy for display (so we don't modify the original if we need it)
        image = frame.copy()

        # flip image (depending on mechanical setup)
        image = cv2.flip(image, -1)

        # plot crosshairs for alignment
        cv2.line(image, (640, 0), (640, 720), (0, 150, 150), 1)
        cv2.line(image, (0, 360), (1280, 360), (0, 150, 150), 1)

        # plot green vertical lines for alignment
        for i in range(50, 1300, 50):
            cv2.line(image, (i, 0), (i, 720), (0, 150, 0), 3)

        # display the image on screen
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # break out of video loop when 'q' is pressed
        if key == ord("q"):
            break

except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    pass

# Cleanup
cv2.destroyAllWindows()
picam2.stop()