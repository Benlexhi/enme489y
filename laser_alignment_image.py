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

# --- ADD THESE CAMERA CONTROLS FOR BETTER QUALITY ---
# These settings lock the camera to manual mode for consistent results
picam2.set_controls({
    "ExposureTime": 20000,  # Exposure in microseconds (20ms = good for indoor)
    "AnalogueGain": 1.0,  # No extra gain (reduces noise)
    "AwbEnable": 0,  # Disable auto white balance (0 = off, 1 = auto)
    "ColourGains": (1.5, 1.5),  # Manual white balance (red gain, blue gain)
    "Contrast": 1.2,  # Slightly increased contrast
    "Brightness": 0.1,  # Slight brightness adjustment
    "Saturation": 1.2  # Boost color saturation a bit
})
# ---------------------------------------------------

# Start the camera
picam2.start()

# allow the camera to warm up with new settings
time.sleep(2)  # Slightly longer to let manual settings stabilize

# Main video loop
try:
    while True:
        # Grab a frame using the "main" stream
        frame = picam2.capture_array("main")

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Rotate image 90 degrees clockwise
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

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
    pass

# Cleanup
cv2.destroyAllWindows()
picam2.stop()