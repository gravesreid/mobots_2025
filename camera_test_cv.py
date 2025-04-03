#!/usr/bin/env python3

import cv2
import time
import os
from datetime import datetime
from headless_visualization import SimpleStreamServer

def main():
    # Create the data/images directory if it doesn't exist
    save_dir = "data/images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Images will be saved to {save_dir}")
    
    # Start the stream server
    server = SimpleStreamServer(port=8080)
    print("Stream Server started")
    
    # Open the RealSense camera using OpenCV (camera index 4)
    cap = cv2.VideoCapture(4)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        server.stop()
        return
    
    print("Camera opened successfully")
    
    try:
        # Initialize frame counter for statistics
        frame_count = 0
        start_time = time.time()
        last_stats_time = start_time
        
        # Main loop
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Update the frame in the web server
            server.update_frame(frame)
            
            # Save the frame with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_mmm
            img_path = os.path.join("data/images", f"image_{timestamp}.jpg")
            # cv2.imwrite(img_path, frame)
            
            # Update statistics
            frame_count += 1
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                fps = frame_count / (current_time - last_stats_time)
                print(f"Streaming at {fps:.2f} FPS, saved {frame_count} images")
                frame_count = 0
                last_stats_time = current_time
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        server.stop()
        print("Resources released and server stopped")

if __name__ == "__main__":
    main()