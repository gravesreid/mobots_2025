#!/usr/bin/env python3

import cv2
import os
from datetime import datetime

def capture_camera_frame():
    # Create a directory to save images if it doesn't exist
    save_dir = "camera_captures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try different camera indices
    # Usually 0 is the first camera, but sometimes the RealSense might be on a different index
    for camera_index in range(3):  # Try indices 0, 1, and 2
        print(f"Trying camera at index {camera_index}...")
        
        # Open the camera
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print(f"Failed to open camera at index {camera_index}")
            continue
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Capture a few frames to allow camera to adjust (exposure, white balance, etc.)
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
        
        # Capture frame
        ret, frame = cap.read()
        
        # Release the camera
        cap.release()
        
        # Check if frame was captured successfully
        if not ret:
            print(f"Failed to capture image from camera at index {camera_index}")
            continue
        
        # Save the image
        image_path = f"{save_dir}/image_{camera_index}_{timestamp}.png"
        cv2.imwrite(image_path, frame)
        
        print(f"Image captured from camera index {camera_index} and saved to: {image_path}")
        return True
    
    print("Failed to capture image from any camera")
    return False

if __name__ == "__main__":
    try:
        result = capture_camera_frame()
        if result:
            print("Frame captured and saved successfully!")
        else:
            print("Failed to capture frame from any camera")
    except Exception as e:
        print(f"An error occurred: {e}")