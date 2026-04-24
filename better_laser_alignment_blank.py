# ENME489Y: Remote Sensing

from picamera2 import Picamera2
import numpy as np
import time
import cv2
import os

# Create a folder for your images
save_folder = "measurements"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={"FrameRate": 25}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# Measurement settings
start_distance = 0  # inches
end_distance = 120  # inches (10 feet)
increment = 6  # inches
distances = list(range(start_distance, end_distance + 1, increment))
# distances = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

current_index = 0
total_measurements = len(distances)

print("\n=== Measurement Camera ===")
print(f"Will take {total_measurements} measurements:")
print(f"Distances: {distances}")
print("\nControls:")
print("  's' - Save current image (auto-named with distance)")
print("  'b' - Go back to previous measurement")
print("  'q' - Quit and show summary")
print("========================\n")

print(f"Ready for measurement {current_index + 1}/{total_measurements}")
print(f"Current distance: {distances[current_index]} inches")

try:
    while current_index < total_measurements:
        # Grab frame
        frame = picam2.capture_array("main")
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, -1)

        # Draw crosshairs
        cv2.line(image, (640, 0), (640, 720), (0, 150, 150), 1)
        cv2.line(image, (600, 360), (1280, 360), (0, 150, 150), 1)

        # Display current distance on screen (large, easy to read)
        current_dist = distances[current_index]
        cv2.putText(image, f"{current_dist} inches", (450, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)

        # Display progress
        cv2.putText(image, f"Measurement: {current_index + 1}/{total_measurements}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display instructions
        cv2.putText(image, "Press 's': Save | 'b': Back | 'q': Quit", (50, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Measurement Camera", image)
        key = cv2.waitKey(1) & 0xFF

        # 's' - Save current image
        if key == ord('s'):
            filename = f"{save_folder}/{current_dist}inches.jpg"
            cv2.imwrite(filename, image)
            print(f"[✓] Saved: {filename}")

            # Move to next measurement
            current_index += 1

            if current_index < total_measurements:
                print(f"\nReady for measurement {current_index + 1}/{total_measurements}")
                print(f"Move camera to {distances[current_index]} inches from object")
            else:
                print("\n🎉 All measurements complete!")
                break

        # 'b' - Go back to previous measurement (in case of mistake)
        elif key == ord('b'):
            if current_index > 0:
                current_index -= 1
                print(f"\nWent back to measurement {current_index + 1}/{total_measurements}")
                print(f"Current distance: {distances[current_index]} inches")
            else:
                print("Already at first measurement, cannot go back")

        # 'q' - Quit early
        elif key == ord('q'):
            print(f"\nQuit early. Completed {current_index}/{total_measurements} measurements")
            break

except KeyboardInterrupt:
    print("\nStopped by user")

# Cleanup
cv2.destroyAllWindows()
picam2.stop()

# Print summary
print("\n" + "=" * 50)
print("MEASUREMENT SUMMARY")
print("=" * 50)
for i in range(current_index):
    dist = distances[i]
    filename = f"{save_folder}/{dist}inches.jpg"
    print(f"{i + 1}. {dist} inches -> {filename}")

print(f"\nCompleted {current_index}/{total_measurements} measurements")