import pyrealsense2 as rs
import numpy as np
import cv2
import time
import argparse

# Import the SimpleStreamServer from our provided code
from headless_visualization import SimpleStreamServer

def main():
    # Create and configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the RGB stream (color)
    # Common resolutions: 640x480, 1280x720
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start the server for web streaming
    server = SimpleStreamServer(port=8080)
    print("RealSense Stream Server started")
    
    try:
        # Start the RealSense pipeline
        pipeline.start(config)
        print("RealSense camera started")
        
        # Initialize stats variables
        frame_count = 0
        start_time = time.time()
        last_stats_time = start_time
        
        # Main loop
        while True:
            # Wait for a coherent set of frames from the camera
            frames = pipeline.wait_for_frames()
            
            # Get the color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert the color frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Update the frame in the web server
            server.update_frame(color_image)
            
            # Update statistics
            frame_count += 1
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:  # Show stats every 5 seconds
                fps = frame_count / (current_time - last_stats_time)
                print(f"Streaming at {fps:.2f} FPS ({frame_count} frames in {current_time - last_stats_time:.1f}s)")
                frame_count = 0
                last_stats_time = current_time
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        pipeline.stop()
        server.stop()
        print("Resources released and server stopped")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RealSense camera streaming over HTTP')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    
    args = parser.parse_args()
    
    main()