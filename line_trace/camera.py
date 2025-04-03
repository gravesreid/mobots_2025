#!/usr/bin/env python3

import cv2
import time
import os
import numpy as np
from datetime import datetime
import sys
sys.path.append("..")
from headless_visualization import SimpleStreamServer
from control import Control

def main():
    control = Control()
    # Start the stream server
    server = SimpleStreamServer(port=8080)
    

    cap = cv2.VideoCapture(4)

    
    if not cap.isOpened():
        print("Error: Could not open any camera")
        server.stop()
        return
    
    print("Camera opened successfully")
    
    try:
        # Main loop
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break

            # ---------- Image processing steps ----------
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur
            blurred = cv2.medianBlur(gray, 7)
            
            # Apply different levels of blurring
            very_blurred = cv2.medianBlur(blurred, 21)
            very_very_blurred = cv2.medianBlur(blurred, 251)
            
            # Take minimum of blurred images
            combo = cv2.min(very_blurred, very_very_blurred)
            very_blurred = combo
            
            # Compute normalized difference
            diff = np.float32(blurred)/np.float32(very_blurred)
            diff = np.uint8(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))
            
            # Apply Otsu thresholding
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological closing
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Keep largest contiguous area
            # Find all contiguous regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
            
            # Find the largest component by area (excluding background at index 0)
            largest_mask = np.zeros_like(closed_mask)
            
            if num_labels > 1:
                largest_label = 1
                largest_area = stats[1, cv2.CC_STAT_AREA]
                
                for i in range(2, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > largest_area:
                        largest_area = area
                        largest_label = i
                
                # Create a new mask containing only the largest component
                largest_mask[labels == largest_label] = 255
            
            # ---------- Prepare visualization of original and mask only ----------
            # Convert mask to BGR for display
            largest_mask_bgr = cv2.cvtColor(largest_mask, cv2.COLOR_GRAY2BGR)
            
            # Make sure both images have the same height
            height, width = frame.shape[:2]
            mask_height, mask_width = largest_mask_bgr.shape[:2]
            
            # Resize mask if needed
            if height != mask_height:
                largest_mask_bgr = cv2.resize(largest_mask_bgr, (int(mask_width * height / mask_height), height))

            if largest_mask_bgr is not None and np.any(largest_mask_bgr):
                # Draw the outline curves
                result_img, left_curve, right_curve = control.draw_outline_curves(largest_mask_bgr, largest_mask_bgr.copy())

                # Create center line
                center_line_image, center_line_top, center_line_bottom, center_line_angle = control.create_center_line(left_curve, right_curve, frame)
                frame = center_line_image  # Use the image with the center line drawn

            
            
            # Create side-by-side display
            combined = np.hstack((frame, largest_mask_bgr))
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(combined, "Contiguous Mask", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Update the frame in the web server
            server.update_frame(combined)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        server.stop()
        print("Resources released and server stopped")

if __name__ == "__main__":
    main()