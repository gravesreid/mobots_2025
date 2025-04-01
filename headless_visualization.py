#!/usr/bin/env python3

import cv2
import time
import threading
import socket
import os
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from datetime import datetime

class StreamHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.server_instance = SimpleStreamServer.instance
        super().__init__(*args, **kwargs)
        
    def log_message(self, format, *args):
        # Silence log messages
        return
    
    def do_GET(self):
        try:
            # Serve index page
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(self._get_index_html().encode('utf-8'))
            
            # Serve video stream
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '-1')
                self.end_headers()
                
                last_frame_time = 0
                
                try:
                    while True:
                        # Only process if we have a frame and it's newer than what we last sent
                        current_frame_time = self.server_instance.frame_timestamp
                        
                        if (self.server_instance.latest_frame is not None and 
                            current_frame_time > last_frame_time):
                            
                            # Update our last sent frame time
                            last_frame_time = current_frame_time
                            
                            # Encode frame as JPEG
                            _, jpeg = cv2.imencode('.jpg', self.server_instance.latest_frame)
                            frame_data = jpeg.tobytes()
                            
                            # Send frame
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(frame_data)}\r\n\r\n'.encode())
                            self.wfile.write(frame_data)
                            self.wfile.write(b'\r\n')
                        
                        time.sleep(0.001)  # Check for new frames frequently
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected
                    pass
            
            # Take a snapshot
            elif self.path == '/snapshot':
                if self.server_instance.latest_frame is not None:
                    # Save the frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-4]
                    filename = f"{self.server_instance.save_dir}/snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, self.server_instance.latest_frame)
                    
                    # Redirect back to the main page
                    self.send_response(303)  # See Other
                    self.send_header('Location', '/')
                    self.end_headers()
                    
                    print(f"Snapshot saved: {filename}")
                else:
                    # No frame available
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'No frame available')
            
            # Serve favicon (to avoid 404 errors in browser)
            elif self.path == '/favicon.ico':
                self.send_response(204)  # No content
                self.end_headers()
            
            # 404 for any other path
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Not found')
        
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected
            pass
        except Exception as e:
            print(f"Error handling request: {e}")
    
    def _get_index_html(self):
        """Generate the HTML for the index page"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pi Display Stream</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    text-align: center;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                }}
                .button {{
                    display: inline-block;
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    margin: 10px 2px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                    font-size: 16px;
                }}
                .button:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Pi Display Stream</h1>
                <img src="/stream" alt="Time Stream">
                <br>
                <a href="/snapshot" class="button">Take Snapshot</a>
            </div>
        </body>
        </html>
        """

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

class SimpleStreamServer:
    def __init__(self, port=8080):
        self.port = port
        self.latest_frame = None
        self.frame_timestamp = 0  # Track when frames are updated
        self.server = None
        self.is_running = False
        self.frame_lock = threading.Lock()  # Thread safety for frame updates
        
        # Create directory for snapshots
        self.save_dir = "snapshots"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Start the server
        self.start()

        self.stopped = False
    
    def start(self):
        """Start the HTTP server in a separate thread"""
        
        # Save instance for handler to access
        SimpleStreamServer.instance = self
        
        # Start server in a new thread
        def run_server():
            server_address = ('', self.port)
            self.server = ThreadedHTTPServer(server_address, StreamHandler)
            self.is_running = True
            
            # Find local IP for user information
            local_ip = self._get_local_ip()
            print(f"\n--------------------------------------------------")
            print(f"Time stream server started at http://{local_ip}:{self.port}")
            print(f"--------------------------------------------------\n")
            
            self.server.serve_forever()
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def update_frame(self, frame):
        """Update the latest frame to be streamed"""
        with self.frame_lock:
            self.latest_frame = frame
            self.frame_timestamp = time.time()  # Record when this frame was created
    
    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            print("Server stopped")
        self.stopped = True
    
    def _get_local_ip(self):
        """Get the local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    def __del__(self):
        if not self.stopped:
            self.stop()
    


def create_time_frame(width=640, height=480):
    """Create a frame displaying the current time with hundredths of seconds"""
    # Create a blank frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get current time with milliseconds
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S.%f")[:-4]  # Format: HH:MM:SS.xx (hundredths)
    date_str = now.strftime("%Y-%m-%d")
    
    # Add background gradient
    for y in range(height):
        # Create a gradient from dark blue to light blue
        blue = int(180 + (y / height) * 75)
        green = int(100 + (y / height) * 100)
        frame[y, :] = [100, green, blue]  # BGR format
    
    # Draw time in large font
    cv2.putText(frame, time_str, (width//2 - 180, height//2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    
    # Draw date in smaller font
    cv2.putText(frame, date_str, (width//2 - 100, height//2 + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Add "Raspberry Pi" label
    cv2.putText(frame, "Raspberry Pi Stream", (width//2 - 140, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Show frame number (useful for detecting missed frames)
    frame_count_str = f"Frame: {create_time_frame.frame_count}"
    cv2.putText(frame, frame_count_str, (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Add a border
    cv2.rectangle(frame, (10, 10), (width-10, height-10), (255, 255, 255), 2)
    
    # Increment frame counter for next call
    create_time_frame.frame_count += 1
    
    return frame

# Initialize the frame counter
create_time_frame.frame_count = 0


def main(interval=0.05, duration=None, no_sleep=False):
    # Start the stream server
    server = SimpleStreamServer(port=8080)
    
    print("Time display stream started")
    print(f"Frame update interval: {interval:.3f}s")
    if no_sleep:
        print("Running in no-sleep mode (maximum frame rate)")
    
    try:
        start_time = time.time()
        frame_count = 0
        last_stats_time = start_time
        
        while True:
            loop_start = time.time()
            
            # Check if duration is set and has elapsed
            if duration and (loop_start - start_time > duration):
                print(f"Duration of {duration}s reached")
                break
            
            # Create a frame with the current time
            frame = create_time_frame()
            
            # Update the stream with the new frame
            server.update_frame(frame)
            
            # Calculate and display stats periodically
            frame_count += 1
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:  # Show stats every 5 seconds
                fps = frame_count / (current_time - last_stats_time)
                print(f"Streaming at {fps:.2f} FPS ({frame_count} frames in {current_time - last_stats_time:.1f}s)")
                frame_count = 0
                last_stats_time = current_time
            
            # Sleep to maintain the requested interval (if not in no_sleep mode)
            if not no_sleep:
                processing_time = time.time() - loop_start
                sleep_time = max(0, interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Clean up
        server.stop()
        print("Server stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Time display streaming server')
    parser.add_argument('--interval', type=float, default=0.05, help='Frame update interval in seconds (default: 0.05)')
    parser.add_argument('--duration', type=float, help='Duration to run in seconds (optional)')
    parser.add_argument('--no-sleep', action='store_true', help='Run at maximum frame rate without sleeping')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    
    args = parser.parse_args()
    
    main(
        interval=args.interval,
        duration=args.duration,
        no_sleep=args.no_sleep
    )