import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import os
import sys

class ImagePathCreator:
    def __init__(self, root, image_path=None):
        self.root = root
        self.root.title("Image Path Creator")
        
        # Initialize variables
        self.image_path = image_path
        self.points = []
        self.binary_mask = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.scale_factor = 1.0
        self.zoom_factor = 1.0
        self.canvas_width = 800
        self.canvas_height = 600
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.dragging = False
        
        # First point hardcoded location (x, y) as specified
        self.first_point = (2418.8, 175.3)
        
        # Create GUI components
        self.create_widgets()
        
        # Load image if provided
        if self.image_path:
            self.load_image()

    def create_widgets(self):
        # Top frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Load image button
        load_btn = tk.Button(button_frame, text="Load PNG Image", command=self.browse_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Save points button
        save_btn = tk.Button(button_frame, text="Save Path Points", command=self.save_points)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear points button
        clear_btn = tk.Button(button_frame, text="Clear Points", command=self.clear_points)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove last point button
        remove_btn = tk.Button(button_frame, text="Remove Last Point", command=self.remove_last_point)
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        # Zoom controls
        zoom_frame = tk.Frame(button_frame)
        zoom_frame.pack(side=tk.LEFT, padx=20)
        
        zoom_in_btn = tk.Button(zoom_frame, text="Zoom In (+)", command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out (-)", command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=5)
        
        reset_zoom_btn = tk.Button(zoom_frame, text="Reset View", command=self.reset_view)
        reset_zoom_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for image display
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.mouse_wheel)    # Linux scroll down
        self.canvas.bind("<B3-Motion>", self.drag_canvas)   # Right button drag for panning
        self.canvas.bind("<ButtonPress-3>", self.start_drag)
        self.canvas.bind("<ButtonRelease-3>", self.stop_drag)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("No image loaded")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Instructions
        instructions = """
        Instructions:
        - Left click to add points
        - Right click and drag to pan
        - Mouse wheel to zoom
        - Use 'Remove Last Point' to undo the last addition
        - Points are saved in order
        """
        instruction_label = tk.Label(self.root, text=instructions, justify=tk.LEFT, anchor=tk.W)
        instruction_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select PNG Image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.load_image()
    
    def load_image(self):
        try:
            # Reset view parameters
            self.reset_view()
            
            # Load the image with PIL (to handle transparency)
            pil_img = Image.open(self.image_path)
            
            # Convert to numpy array
            np_img = np.array(pil_img)
            
            # Create binary mask from alpha channel
            if np_img.shape[2] == 4:  # Check if image has alpha channel
                # Set 1 for non-transparent pixels, 0 for transparent
                alpha = np_img[:, :, 3]
                self.binary_mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.uint8)
                self.binary_mask[alpha > 0] = 1
                
                # Create a white background image
                white_bg = np.ones((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8) * 255
                
                # Create a mask for blending
                mask = np.stack([alpha, alpha, alpha], axis=2) / 255.0
                
                # Blend the image with white background where transparent
                rgb_img = np_img[:, :, :3]
                self.original_image = (rgb_img * mask + white_bg * (1 - mask)).astype(np.uint8)
            else:
                # If no alpha channel, use the image as is
                self.binary_mask = np.ones((np_img.shape[0], np_img.shape[1]), dtype=np.uint8)
                self.original_image = np_img[:, :, :3]  # Just take RGB channels
            
            # Keep the hardcoded first point (2418.8, 157.3) - don't override it
            
            # Reset points and add first point
            self.points = [self.first_point]
            
            # Calculate initial scale factor to fit the image in the canvas
            h, w = self.original_image.shape[:2]
            width_scale = self.canvas_width / w
            height_scale = self.canvas_height / h
            self.scale_factor = min(width_scale, height_scale)
            
            # Display the image
            self.update_display()
            
            # Update status
            self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)} | Mask created | First point set at {self.first_point}")
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
    
    def update_display(self):
        if self.original_image is None:
            return
        
        # Make a copy of the original image
        display_img = self.original_image.copy()
        
        # Calculate the effective scale (base scale * zoom)
        effective_scale = self.scale_factor * self.zoom_factor
        
        # Resize for display based on scale and zoom
        h, w = display_img.shape[:2]
        new_h = int(h * effective_scale)
        new_w = int(w * effective_scale)
        
        if new_h > 0 and new_w > 0:  # Ensure positive dimensions
            display_img = cv2.resize(display_img, (new_w, new_h))
            
            # Draw points and lines
            for i, point in enumerate(self.points):
                # Scale point according to display - convert float coordinates to integers
                scaled_x = int(float(point[0]) * effective_scale)
                scaled_y = int(float(point[1]) * effective_scale)
                
                # Draw point
                cv2.circle(display_img, (scaled_x, scaled_y), max(3, int(5 * effective_scale)), (0, 0, 255), -1)
                
                # Draw number
                font_scale = max(0.3, 0.5 * effective_scale)
                cv2.putText(display_img, str(i), (scaled_x + 10, scaled_y + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                
                # Draw line between points
                if i > 0:
                    prev_point = self.points[i-1]
                    scaled_prev_x = int(prev_point[0] * effective_scale)
                    scaled_prev_y = int(prev_point[1] * effective_scale)
                    cv2.line(display_img, (scaled_prev_x, scaled_prev_y), 
                             (scaled_x, scaled_y), (255, 0, 0), max(1, int(2 * effective_scale)))
            
            # Convert to PIL format
            self.display_image = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            self.tk_image = ImageTk.PhotoImage(image=self.display_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
            self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image, tags="image")
    
    def canvas_click(self, event):
        if self.original_image is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Adjust for panning
        canvas_x -= self.pan_x
        canvas_y -= self.pan_y
        
        # Convert canvas coordinates to original image coordinates
        effective_scale = self.scale_factor * self.zoom_factor
        original_x = int(canvas_x / effective_scale)
        original_y = int(canvas_y / effective_scale)
        
        # Add point to list (as floats for precision)
        self.points.append((float(original_x), float(original_y)))
        
        # Update display
        self.update_display()
        
        # Update status
        self.status_var.set(f"Added point {len(self.points)-1} at ({original_x}, {original_y}) | Total points: {len(self.points)}")
    
    def start_drag(self, event):
        self.dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def stop_drag(self, event):
        self.dragging = False
    
    def drag_canvas(self, event):
        if self.dragging and self.original_image is not None:
            # Calculate the movement
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            # Update the pan
            self.pan_x += dx
            self.pan_y += dy
            
            # Update drag start position
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            
            # Update the display
            self.update_display()
    
    def mouse_wheel(self, event):
        if self.original_image is None:
            return
        
        # Determine the direction of scroll
        if event.num == 4 or event.delta > 0:  # Scroll up (zoom in)
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:  # Scroll down (zoom out)
            self.zoom_out()
    
    def zoom_in(self):
        if self.original_image is not None:
            self.zoom_factor *= 1.2
            self.update_display()
            self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
    
    def zoom_out(self):
        if self.original_image is not None:
            if self.zoom_factor > 0.2:  # Limit how far we can zoom out
                self.zoom_factor /= 1.2
                self.update_display()
                self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
    
    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        if self.original_image is not None:
            self.update_display()
            self.status_var.set("View reset")
    
    def clear_points(self):
        if self.original_image is None:
            return
        
        # Keep only first point
        self.points = [self.first_point]
        
        # Update display
        self.update_display()
        
        # Update status
        self.status_var.set(f"Points cleared. First point at {self.first_point}")
        
    def remove_last_point(self):
        if self.original_image is None:
            return
            
        # Check if we have more than just the first point
        if len(self.points) > 1:
            # Remove the last point
            removed_point = self.points.pop()
            
            # Update display
            self.update_display()
            
            # Update status
            self.status_var.set(f"Removed point at ({removed_point[0]}, {removed_point[1]}) | Total points: {len(self.points)}")
        else:
            self.status_var.set("Cannot remove the first point")
    
    def save_points(self):
        if not self.points:
            self.status_var.set("No points to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Path Points",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["point_id", "x", "y"])
                    for i, (x, y) in enumerate(self.points):
                        writer.writerow([i, x, y])
                
                self.status_var.set(f"Saved {len(self.points)} points to {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error saving points: {str(e)}")

def run_app():
    root = tk.Tk()
    app = ImagePathCreator(root)
    root.geometry("900x700")
    root.mainloop()


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from scipy.spatial.distance import euclidean

def process_path(csv_file, output_file, pixel_to_meter=0.03, spacing=0.05, smoothness=0.5, show_plot=True):
    """
    Load points from CSV, convert to meters, fit a spline, and generate equidistant points.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the points
    output_file : str
        Path to save the processed points
    pixel_to_meter : float, default=0.03
        Conversion factor from pixels to meters (1 pixel = 0.03 meters)
    spacing : float, default=0.05
        Desired spacing between points in meters
    smoothness : float, default=0.1
        Smoothing factor for the spline:
        0 = exact interpolation (passes through all points)
        Larger values = smoother curve that may not pass through all points
    show_plot : bool, default=True
        Whether to display a plot of the original points and the spline
        
    Returns:
    --------
    numpy.ndarray
        Array of processed points with equidistant spacing
    """
    # Load the points from CSV using numpy
    points = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            points.append([float(row[1]), float(row[2])])  # x, y columns
    
    points = np.array(points)
    
    # Extract the initial point to use as origin
    origin = points[0]
    print(f"Origin point (pixels): ({origin[0]}, {origin[1]})")
    
    # Convert to meters with origin at initial point
    # Using the experimentally verified transformation
    points_meters = np.zeros_like(points, dtype=float)
    for i, point in enumerate(points):
        # Subtract origin and apply conversion with correct transformation
        points_meters[i, 0] = -1 * (point[0] - origin[0]) * pixel_to_meter  # Negative X transformation
        points_meters[i, 1] = 1 * (point[1] - origin[1]) * pixel_to_meter   # Positive Y transformation
    
    # The first point should now be (0, 0) in the new coordinate system
    print(f"First point after conversion (meters): ({points_meters[0, 0]:.6f}, {points_meters[0, 1]:.6f})")
    
    # Separate x and y coordinates
    x_meters = points_meters[:, 0]
    y_meters = points_meters[:, 1]
    
    # Compute cumulative distance along the path
    distances = [0]
    for i in range(1, len(points_meters)):
        d = euclidean(points_meters[i-1], points_meters[i])
        distances.append(distances[-1] + d)
    
    distances = np.array(distances)
    
    # Create a parametric spline using cumulative distance as parameter
    # Use UnivariateSpline with direct smoothing parameter
    s_x = UnivariateSpline(distances, x_meters, s=smoothness)
    s_y = UnivariateSpline(distances, y_meters, s=smoothness)
    
    print(f"Spline created with smoothness factor: {smoothness}")
    
    # Total path length
    path_length = distances[-1]
    print(f"Total path length: {path_length:.2f} meters")
    
    # Number of points needed for the desired spacing
    num_points = int(np.ceil(path_length / spacing)) + 1
    
    # Generate equidistant points along the spline
    t_values = np.linspace(0, path_length, num_points)
    equidistant_points = np.zeros((num_points, 2))
    equidistant_points[:, 0] = s_x(t_values)
    equidistant_points[:, 1] = s_y(t_values)
    
    # Save the processed points
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x_meters", "y_meters"])
        for i, point in enumerate(equidistant_points):
            writer.writerow([point[0], point[1]])
    
    print(f"Saved {len(equidistant_points)} processed points to {output_file}")
    
    # Visualize the results
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot original points
        plt.plot(x_meters, y_meters, 'bo', label='Original Points')
        
        # Plot spline with more points for smoother visualization
        t_fine = np.linspace(0, path_length, 1000)
        plt.plot(s_x(t_fine), s_y(t_fine), 'g-', label='Fitted Spline')
        
        # Plot equidistant points
        plt.plot(equidistant_points[:, 0], equidistant_points[:, 1], 'rx', label='Equidistant Points (0.05m)')
        
        plt.title('Path with Equidistant Points')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return equidistant_points

# Main execution
if __name__ == "__main__":
    # ====== EDIT THESE VARIABLES AS NEEDED ======
    # Input/output files
    INPUT_CSV_FILE = "map_processing/test_points_pix.csv"  # Change to your input CSV file path
    OUTPUT_CSV_FILE = "map_processing/test_points_processed.csv"  # Change to your desired output file path

    # Conversion parameters
    PIXEL_TO_METER = 0.03  # Conversion factor: 1 pixel = 0.03 meters
    POINT_SPACING = 0.05   # Desired spacing between points in meters
    SMOOTHNESS = 0.05        # Smoothing factor: 0 = exact interpolation, larger values = smoother curve
    SHOW_PLOT = True       # Set to False to disable plotting
    # ============================================
    process_path(
        csv_file=INPUT_CSV_FILE,
        output_file=OUTPUT_CSV_FILE,
        pixel_to_meter=PIXEL_TO_METER,
        spacing=POINT_SPACING,
        smoothness=SMOOTHNESS,
        show_plot=SHOW_PLOT
    )
