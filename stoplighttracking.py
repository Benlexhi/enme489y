# ENME 489Y: Remote Sensing

# Python script tracks green 'stoplight'
# and saves video of tracking to stoplight.mp4

# import the necessary packages
from picamera2 import Picamera2
import numpy as np
import cv2
import time

# define the lower and upper boundaries of the
# green circle in the HSV color space
colorLower = (45, 100, 100)
colorUpper = (75, 255, 255)

# initialize the Raspberry Pi camera with picamera2
picam2 = Picamera2()

# Configure camera settings
config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": 25}
)
picam2.configure(config)
picam2.start()

# Setup OpenCV VideoWriter for MP4 (records what's on screen)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# For AVI format, use: fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stoplight_tracking.mp4', fourcc, 25.0, (640, 480))
print("Recording to stoplight_tracking.mp4")

# Allow camera to warmup
time.sleep(0.1)

# keep looping
try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Convert from RGB to BGR (OpenCV uses BGR)
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # blur the frame and convert to the HSV color space
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green"
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        center = None

        # proceed regardless to keep video streaming
        if len(cnts) > 0:
            # find the largest contour
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 10:  # Only track significant objects
                    # draw the circle and centroid on the frame
                    cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(image, center, 2, (0, 0, 255), -1)

        # WRITE THE ANNOTATED FRAME TO VIDEO (this captures the circles!)
        out.write(image)

        # show the frame to our screen
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # press the 'q' key to stop
        if key == ord("q"):
            print("Stopping recording...")
            break

finally:
    # Clean up
    out.release()  # Save and close the video file
    picam2.stop()
    cv2.destroyAllWindows()
    print("Video saved as stoplight_tracking.mp4")