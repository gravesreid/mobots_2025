#!/usr/bin/env python3

import cv2
import os
import time
from datetime import datetime

def capture_timelapse(camera_index=4, interval=0.1, duration=10):
    """
    Capture images at specified intervals using the camera at the given index.
    
    Args:
        camera_index (int): Index of the camera to use (default is 4)
        interval (float): Time interval between captures in seconds (default is 0.1)
        duration (int): Total duration to capture in seconds (default is 10)
    """
    # Create a directory to save images if it doesn't exist
    save_dir = "timelapse_captures"
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp_dir)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Opening camera at index {camera_index}...")
    
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Failed to open camera at index {camera_index}")
        return False
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm up the camera
    print("Warming up camera...")
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed during camera warm-up")
            cap.release()
            return False
    
    try:
        frame_count = 0
        start_time = time.time()
        print(f"Starting capture for {duration} seconds with {interval}s intervals")
        
        # Calculate total expected frames
        expected_frames = int(duration / interval)
        
        while time.time() - start_time < duration:
            # Capture frame
            ret, frame = cap.read()
            
            # Check if frame was captured successfully
            if not ret:
                print("Failed to capture frame, skipping...")
                continue
            
            # Save the image
            frame_count += 1
            image_path = os.path.join(save_path, f"frame_{frame_count:04d}.png")
            cv2.imwrite(image_path, frame)
            
            # Print progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Captured {frame_count}/{expected_frames} frames, elapsed time: {elapsed:.2f}s")
            
            # Calculate time to wait until next capture
            capture_end = time.time()
            next_capture_time = start_time + (frame_count * interval)
            
            # Wait until next capture time
            sleep_time = max(0, next_capture_time - capture_end)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Processing taking longer than interval ({-sleep_time:.4f}s behind)")
        
        total_time = time.time() - start_time
        print(f"Capture complete! {frame_count} frames captured over {total_time:.2f} seconds")
        print(f"Images saved to: {save_path}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\nCapture interrupted! {frame_count} frames were saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
        
    finally:
        # Release the camera
        cap.release()
        print("Camera released")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Capture a timelapse sequence from a camera')
    parser.add_argument('--camera', type=int, default=4, help='Camera index (default: 4)')
    parser.add_argument('--interval', type=float, default=0.1, help='Capture interval in seconds (default: 0.1)')
    parser.add_argument('--duration', type=int, default=10, help='Total capture duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Call the capture function with command-line arguments
    capture_timelapse(
        camera_index=args.camera,
        interval=args.interval,
        duration=args.duration
    )