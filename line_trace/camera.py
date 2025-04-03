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

# Use global variables for line state information
found_line = False
center_angle = 0.0  # Add a variable to store the center line angle

def main():
    global found_line, center_angle
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
            frame = cv2.resize(frame, (480, 270))

            
            if not ret:
                print("Error: Could not read frame")
                break

            # Reset found_line at the beginning of each iteration
            found_line = False
            
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
                # Check if too much of the image is white (false detection)
                white_pixel_count = np.sum(largest_mask > 0)
                total_pixels = largest_mask.shape[0] * largest_mask.shape[1]
                white_ratio = white_pixel_count / total_pixels
                
                # If more than 2/3 of the image is white, consider it a false detection
                if white_ratio > 2/3:
                    print(f"False detection - too much white ({white_ratio:.2f})")
                else:
                    try:
                        # Draw the outline curves
                        result_img, left_curve, right_curve = control.draw_outline_curves(largest_mask_bgr, largest_mask_bgr.copy())

                        # Create center line
                        center_line_image, center_line_top, center_line_bottom, center_line_angle = control.create_center_line(left_curve, right_curve, frame)
                        
                        if center_line_top is not None and center_line_bottom is not None:
                            frame = center_line_image  # Use the image with the center line drawn
                            found_line = True
                            center_angle = center_line_angle  # Store the center line angle
                            print(f"Line found! Angle: {center_angle:.2f}°")
                    except Exception as e:
                        print(f"Error processing line: {e}")

            # Create side-by-side display
            combined = np.hstack((frame, largest_mask_bgr))
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(combined, "Contiguous Mask", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add angle information
            if found_line:
                angle_text = f"Line Angle: {center_angle:.2f}°"
                cv2.putText(combined, angle_text, (10, height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
                
                # Add steering direction indicator
                if center_angle > 0:
                    direction = "Steering Right"
                elif center_angle < 0:
                    direction = "Steering Left"
                else:
                    direction = "Straight"
                    
                cv2.putText(combined, direction, (width // 2 - 80, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(combined, "No Line Detected", (10, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Update the frame in the web server
            server.update_frame(combined)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        server.stop()
        print("Resources released and server stopped")

def get_line_state():
    """Function to access line found status and center line angle from outside this module"""
    global found_line, center_angle
    return found_line, center_angle

if __name__ == "__main__":
    main()