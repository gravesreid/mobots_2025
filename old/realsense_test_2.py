import pyrealsense2 as rs
import numpy as np
import time

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream the color stream
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

# Start streaming
try:
    print("Starting camera...")
    pipeline.start(config)
    print("Camera started successfully!")
    
    # Get 30 frames and report
    for i in range(30):
        print(f"Waiting for frame {i+1}/30...")
        frames = pipeline.wait_for_frames(10000)  # 10 second timeout
        color_frame = frames.get_color_frame()
        
        if color_frame:
            # Get frame data
            frame_data = np.asanyarray(color_frame.get_data())
            print(f"Received frame {i+1}: {frame_data.shape} - min:{frame_data.min()} max:{frame_data.max()}")
        else:
            print(f"Frame {i+1}: No color data")
            
        time.sleep(0.1)
        
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    pipeline.stop()
    print("Pipeline stopped")