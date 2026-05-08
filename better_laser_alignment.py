#This one doesn't work as well

# ENME 489Y: Remote Sensing
# Assignment 5: Alignment of field deployable lidar using line laser

# Import packages
import numpy as np
import argparse
import cv2
import imutils
import glob
import re
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import os

print("All packages imported properly!")

# Create debug folder if it doesn't exist
debug_folder = "debug_images"
if not os.path.exists(debug_folder):
    os.makedirs(debug_folder)
    print(f"Created debug folder: {debug_folder}")

files = glob.glob('measurements/*.jpg')  # finds all the pathnames matching a specified pattern
print(f"Found {len(files)} images: {files}")

# define the lower and upper boundaries of the
# red line in the HSV color space
colorLower = (163, 97, 16) #42
colorUpper = (193, 255, 255)

# initialize plot arrays
x_plot = []
y_plot = []


def detect_laser_line_hough(image, mask):
    """
    Detect the laser line using Hough Transform
    Returns the x-coordinate where the line crosses the middle row (y=360)
    """
    # Apply mask to get only red regions
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=50)

    if lines is not None:
        # Find the line that best fits the laser (usually the longest or most vertical)
        best_line = None
        best_length = 0
        best_x_intersect = None

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip lines that are too short
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 50:
                continue

            # Skip vertical lines (x1 == x2) to avoid division by zero
            if x2 == x1:
                # Vertical line - it will intersect at x = x1 for all y
                x_intersect = x1
                if 0 <= x_intersect < image.shape[1]:
                    if length > best_length:
                        best_length = length
                        best_line = (x1, y1, x2, y2)
                        best_x_intersect = x_intersect
                continue

            # Calculate slope and intercept for non-vertical lines
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Check if m is valid (not NaN or infinite)
            if np.isnan(m) or np.isinf(m):
                continue

            # Solve for x when y = 360
            x_intersect = int((360 - b) / m)

            # Check if intersection is within image bounds
            if 0 <= x_intersect < image.shape[1]:
                # Prefer longer lines (more likely to be the laser)
                if length > best_length:
                    best_length = length
                    best_line = (x1, y1, x2, y2)
                    best_x_intersect = x_intersect

        if best_x_intersect is not None:
            return best_x_intersect, best_line

    return None, None


def detect_laser_line_intensity(image, search_side='right'):
    """
    Fallback method: Find brightest pixels in the middle row
    search_side: 'left', 'right', or 'center'
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look at the middle row
    y = 360
    if y >= gray.shape[0]:
        y = gray.shape[0] // 2

    row = gray[y, :]

    # Restrict search to specific side if needed
    if search_side == 'right':
        row = row[gray.shape[1] // 2:]  # Only right half
        offset = gray.shape[1] // 2
    elif search_side == 'left':
        row = row[:gray.shape[1] // 2]  # Only left half
        offset = 0
    else:
        offset = 0

    # Find the brightest pixel
    max_val = np.max(row)
    if max_val > 100:  # Threshold for minimum brightness
        x_pos = np.argmax(row) + offset
        return x_pos

    return None


def enhance_laser_visibility(image):
    """
    Enhance the image to make the laser more visible
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


for idx, filename in enumerate(files):  # x is the filename
    print(f"\n{'=' * 50}")
    print(f"Processing {idx + 1}/{len(files)}: {filename}")
    print(f"{'=' * 50}")

    # Extract distance from filename
    match = re.search(r'(\d+)', filename)
    distance_val = int(match.group(1)) if match else 0
    print(f"Distance: {distance_val} inches")

    # Load image
    image = cv2.imread(filename)
    if image is None:
        print(f"  Could not load image: {filename}")
        continue

    # Enhance image for better laser visibility
    enhanced = enhance_laser_visibility(image)

    # Blur and convert to HSV
    blurred = cv2.GaussianBlur(enhanced, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Construct mask for red color
    mask = cv2.inRange(hsv, colorLower, colorUpper)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise

    # Try to detect laser line using Hough Transform
    spot, best_line = detect_laser_line_hough(image, mask)

    # If Hough fails, try intensity-based detection (search on the right side)
    if spot is None:
        print("  Hough detection failed, trying intensity method...")
        spot = detect_laser_line_intensity(enhanced, search_side='right')

    # If still no detection, use the brightest pixel in the masked region
    if spot is None:
        print("  Intensity method failed, using max from mask...")
        # Find brightest pixel in the masked region
        masked = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        y = 360
        if y < gray.shape[0]:
            row = gray[y, :]
            if np.max(row) > 50:
                spot = np.argmax(row)
            else:
                spot = 1
                print("  WARNING: No laser detected, using default spot=1")
        else:
            spot = 1

    print(f"  ✅ Detected laser spot at x = {spot}")

    # Draw results on image for debugging
    result_image = image.copy()
    y_line = 360
    if y_line >= result_image.shape[0]:
        y_line = result_image.shape[0] // 2

    # Mark the middle row
    cv2.line(result_image, (0, y_line), (result_image.shape[1], y_line), (0, 255, 0), 2)

    # Mark the detected spot
    cv2.circle(result_image, (spot, y_line), 10, (0, 0, 255), -1)

    # Draw detected line if found
    if best_line is not None:
        x1, y1, x2, y2 = best_line[:4]  # Only take first 4 values
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Add text information to debug image
    cv2.putText(result_image, f"Distance: {distance_val} inches", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f"Spot at x = {spot}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create HSV mask visualization
    # Convert mask to BGR for display
    mask_viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Add overlay to show what the mask detected
    overlay = image.copy()
    # Highlight masked regions in red on the overlay
    overlay[mask == 255] = [0, 0, 255]  # Red overlay where mask is white
    masked_viz = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Add text to mask visualization
    cv2.putText(mask_viz, f"HSV Mask - {distance_val} inches", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(mask_viz, f"White areas = detected red", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Also create a side-by-side comparison
    h, w = image.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = result_image
    comparison[:, w:w * 2] = mask_viz

    # Add separator line
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

    # Add labels
    cv2.putText(comparison, "Original with Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "HSV Mask Result", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "Press any key for next, 'q' to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- DISPLAY the side-by-side comparison ---
    cv2.imshow(f"Laser Detection - {distance_val} inches", comparison)

    # --- SAVE all debug images ---
    # Save the comparison image
    debug_filename = os.path.join(debug_folder, f"debug_{distance_val}inches.jpg")
    cv2.imwrite(debug_filename, comparison)
    print(f"  💾 Saved debug comparison: {debug_filename}")

    # Save individual components
    result_filename = os.path.join(debug_folder, f"result_{distance_val}inches.jpg")
    cv2.imwrite(result_filename, result_image)

    mask_filename = os.path.join(debug_folder, f"mask_{distance_val}inches.jpg")
    cv2.imwrite(mask_filename, mask)

    enhanced_filename = os.path.join(debug_folder, f"enhanced_{distance_val}inches.jpg")
    cv2.imwrite(enhanced_filename, enhanced)

    masked_viz_filename = os.path.join(debug_folder, f"masked_viz_{distance_val}inches.jpg")
    cv2.imwrite(masked_viz_filename, masked_viz)

    print(f"  💾 Saved result image: {result_filename}")
    print(f"  💾 Saved mask: {mask_filename}")
    print(f"  💾 Saved enhanced image: {enhanced_filename}")
    print(f"  💾 Saved masked visualization: {masked_viz_filename}")

    # Write coordinates to file
    with open('laserlog.txt', 'a') as f:
        outstring = f"{distance_val} {spot}\n"
        f.write(outstring)

    x_plot.append(spot)
    y_plot.append(distance_val)

    # Wait for user input before continuing
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(f"Laser Detection - {distance_val} inches")

    if key == ord('q'):
        print("\n⚠️ User quit early")
        break

# Close any remaining windows
cv2.destroyAllWindows()

# Check if we have valid data
if len(x_plot) == 0:
    print("No valid measurements found!")
    exit()

# Define ro and rpc, specific to your lidar
ro = -0.0019  # Radian offset
rpc = 0.00008  # Radians per pixel pitch

# Define pixel range from center
pfc = np.arange(0, 640, 2)

# Separation distance between axes [cm] (12 inches = 0.3048 meters)
H = 0.3048

# Calculate distance to target
try:
    D = H / (np.tan(pfc * rpc + ro))
    D = np.flip(D, 0)
except RuntimeWarning:
    print("Warning: Invalid values in theoretical curve calculation")
    D = np.zeros_like(pfc)

# Graph results
plt.figure(1, figsize=(12, 8))
plt.plot(x_plot, y_plot, 'ro', markersize=8, label='Measured Data')
plt.plot(pfc, D, 'b-', linewidth=3, label='Theoretical Curve')
plt.title('Lidar Calibration Curve', fontsize=14)
plt.xlabel('Pixel Position', fontsize=12)
plt.ylabel('Distance to Target (Inches)', fontsize=12)
plt.axis([0, 640, 0, 120])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Save the plot as an image
#plot_filename = os.path.join(debug_folder, "calibration_plot.png")
#plt.savefig(plot_filename, dpi=150)
#print(f"\n💾 Saved plot to: {plot_filename}")

print("\n" + "=" * 50)
print("PROCESSING COMPLETE!")
print("=" * 50)
print(f"✅ Processed {len(x_plot)} images")
print(f"📁 Debug images saved in: {debug_folder}/")
print(f"   - debug_Xinches.jpg (side-by-side comparison)")
print(f"   - result_Xinches.jpg (original with detection overlays)")
print(f"   - mask_Xinches.jpg (raw HSV mask)")
print(f"   - enhanced_Xinches.jpg (contrast-enhanced images)")
print(f"   - masked_viz_Xinches.jpg (overlay showing detected regions)")
print(f"   - calibration_plot.png (final plot)")
print(f"📊 Data saved in: laserlog.txt")