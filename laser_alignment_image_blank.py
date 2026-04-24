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

# Enter distance from wall, entered by the user
d = input("Please enter distance from wall, in inches: ")
print("Confirming the distance you entered is: ", d)

# Main video loop (replaces capture_continuous)
try:
    while True:
        # Grab a frame from the camera
        frame = picam2.capture_array("main")

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # flip image
        image = cv2.flip(image, -1)

        # plot semi-crosshairs for alignment
        cv2.line(image, (640, 0), (640, 720), (0, 150, 150), 1)
        cv2.line(image, (600, 360), (1280, 360), (0, 150, 150), 1)

        # display distance from the wall, for reference
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        red = (0, 0, 255)  # BGR format - red
        cv2.putText(image, str(d), (800, 200), font, 10, red, 10)

        # display the image on screen
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # press q to break out of video stream
        if key == ord("q"):
            break

        # press m to save .jpg image with distance as filename
        if key == ord("m"):
            # Convert d to int for filename
            distance_num = int(d)
            filename = f"{distance_num}.jpg"
            cv2.imwrite(filename, image)
            print(f"Image saved as {filename}")
            break

except KeyboardInterrupt:
    print("Stopped by user")

# Cleanup
cv2.destroyAllWindows()
picam2.stop()