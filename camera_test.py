#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def capture_realsense_frame():
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config and configure the pipeline to stream
    config = rs.config()
    
    # Enable the streams (both depth and color)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    print("Starting RealSense camera...")
    pipeline_profile = pipeline.start(config)
    
    try:
        # Wait for the camera to warm up
        print("Waiting for camera to warm up...")
        for i in range(30):
            pipeline.wait_for_frames()
        
        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory if it doesn't exist
        save_dir = "realsense_captures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Capture a single frameset
        print("Capturing frames...")
        frameset = pipeline.wait_for_frames()
        
        # Get depth and color frames
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Error: Could not capture frames")
            return False
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply colormap on depth image (convert to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Save the images
        color_path = f"{save_dir}/color_{timestamp}.png"
        depth_path = f"{save_dir}/depth_{timestamp}.png"
        
        cv2.imwrite(color_path, color_image)
        cv2.imwrite(depth_path, depth_colormap)
        
        print(f"Color image saved to: {color_path}")
        print(f"Depth image saved to: {depth_path}")
        
        return True
    
    finally:
        # Stop streaming
        pipeline.stop()
        print("Camera stopped")

if __name__ == "__main__":
    try:
        result = capture_realsense_frame()
        if result:
            print("Frame captured and saved successfully!")
        else:
            print("Failed to capture frame")
    except Exception as e:
        print(f"An error occurred: {e}")