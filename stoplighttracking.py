# ENME 489Y: Remote Sensing

# Python script tracks green 'stoplight'
# and saves video of tracking to stoplight.mp4

# import the necessary packages
from picamera2 import Picamera2
import numpy as np
import imutils
import cv2
import time

# define the lower and upper boundaries of the
# green circle in the HSV color space
# Note: use colorpicker.py to create a new HSV mask
colorLower = (40, 50, 50)
colorUpper = (80, 255, 255)

# initialize the Raspberry Pi camera with picamera2
picam2 = Picamera2()

# Configure camera settings
config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": 25}
)
picam2.configure(config)
picam2.start()

# Allow camera to warmup
time.sleep(0.1)

# define the codec and create VideoWriter object
# UNCOMMENT THE FOLLOWING LINES TO SAVE VIDEO FILE
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('stoplight.avi', fourcc, 10, (640, 480))

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

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        center = None

        # proceed regardless to keep video streaming
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] != 0:  # Avoid division by zero
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 0:
                    # draw the circle and centroid on the frame
                    cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(image, center, 2, (0, 0, 255), -1)

                # write the frame to video file
                # UNCOMMENT THE FOLLOWING LINE TO SAVE VIDEO FILE
                # out.write(image)

        # show the frame to our screen
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # press the 'q' key to stop the video stream
        if key == ord("q"):
            break

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
# out.release()  # Uncomment if using video saving