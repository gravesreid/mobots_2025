import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from math import sin, cos, radians
import sys
from io import StringIO
from contextlib import redirect_stdout
sys.path.append('..')  # Add parent directory to path
from image_transform_reverse import real_coords_to_pixel_coords
from image_transform_reverse2 import get_warped_image


class Control:
    def __init__(self):
        # Use a copy of the original image
        pass

    def draw_outline_curves(self, mask, original_image=None):
        """
        Draw two vertical curves that outline the white object in the mask.
        
        Args:
            mask: Binary mask with the white object
            original_image: Optional original image to draw on (if None, will draw on a copy of the mask)
        
        Returns:
            Tuple of (image with curves, left_curve, right_curve)
        """
        # Use a copy of the original image
        result_img = original_image.copy()
        
        # Get the height and width of the mask
        height, width = mask.shape[:2]
        
        # Initialize lists to store the left and right curve points
        left_curve = []
        right_curve = []
        
        # For each row in the mask
        for y in range(height):
            row = mask[y, :]
            white_pixels = np.where(row > 0)[0]
            
            # If there are white pixels in this row
            if len(white_pixels) > 0:
                # Get the leftmost and rightmost white pixels
                left_x = white_pixels[0]
                right_x = white_pixels[-1]
                
                # Add to the curves
                left_curve.append((left_x, y))
                right_curve.append((right_x, y))
        
        # Convert lists to numpy arrays for drawing
        if left_curve and right_curve:
            left_curve = np.array(left_curve, dtype=np.int32)
            right_curve = np.array(right_curve, dtype=np.int32)
            
            # Draw the curves
            cv2.polylines(result_img, [left_curve], False, (0, 0, 255), 2)  # Red for left curve
            cv2.polylines(result_img, [right_curve], False, (255, 0, 0), 2)  # Blue for right curve
        
        # Return both the image and the curves for further processing
        return result_img, left_curve, right_curve

    def process_image(self, image_path, robot_x=20, robot_y=0, robot_angle=-15):
        """
        Process an image and visualize the robot on the map.
        
        Args:
            image_path: Path to the camera image
            robot_x: Current robot x position (meters)
            robot_y: Current robot y position (meters)
            robot_angle: Current robot heading angle (degrees)
        """
        # Load the image
        img = self.image_processing(image_path)
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Draw outline curves
        outlined_image, left_curve, right_curve = self.draw_outline_curves(gray, img)
        
        # Create and draw center line
        center_line_image, center_line_top, center_line_bottom, center_line_angle = self.create_center_line(left_curve, right_curve, outlined_image)
        
        # Save the outlined image with center line directly (no control/ directory)
        cv2.imwrite("center_line.png", center_line_image)
        print("Image with center line saved to center_line.png")
        
        # Visualize robot on the map
        self.visualize_robot_on_map(
            robot_x, 
            robot_y, 
            robot_angle, 
            center_line_angle=center_line_angle
        )

    def image_processing(self, image_path):
        """
        Load the image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Loaded image
        """
        # Check if the file exists
        if not os.path.exists(image_path):
            # Try looking in the control directory
            control_path = os.path.join("control", image_path)
            if os.path.exists(control_path):
                image_path = control_path
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return img

    def create_center_line(self, left_curve, right_curve, img, start_at_bottom_third=True):
        """
        Create a straight center line based on the left and right curves.
        
        Args:
            left_curve: Array of points [(x1,y1), (x2,y2), ...] for left curve
            right_curve: Array of points for right curve
            img: Image to draw on
            start_at_bottom_third: Whether to only use bottom third of the image
            
        Returns:
            Image with center line drawn
        """
        height, width = img.shape[:2]
        result_img = img.copy()
        
        # Filter points to ensure we have matching y-coordinates
        left_dict = {point[1]: point[0] for point in left_curve}
        right_dict = {point[1]: point[0] for point in right_curve}
        
        # Find common y-coordinates
        common_y = set(left_dict.keys()).intersection(set(right_dict.keys()))
        
        # For bottom third of the image
        bottom_third_y = height * 2 // 3
        if start_at_bottom_third:
            common_y = [y for y in common_y if y >= bottom_third_y]
        
        # Calculate midpoints
        midpoints = []
        for y in common_y:
            left_x = left_dict[y]
            right_x = right_dict[y]
            mid_x = (left_x + right_x) // 2
            midpoints.append((mid_x, y))
        
        if len(midpoints) < 2:
            print("Not enough points to create a center line")
            return result_img, None, None, None
        
        # Convert to numpy array
        midpoints = np.array(midpoints)
        
        # Fit a straight line using linear regression
        x = midpoints[:, 0]
        y = midpoints[:, 1]
        
        # Use polyfit to get the line parameters (degree 1 = straight line)
        try:
            slope, intercept = np.polyfit(y, x, 1)
            
            # Create the line endpoints
            if start_at_bottom_third:
                top_y = bottom_third_y
            else:
                top_y = 0
            bottom_y = height - 1
            
            top_x = int(slope * top_y + intercept)
            bottom_x = int(slope * bottom_y + intercept)
            
            # Draw the line
            cv2.line(result_img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 3)  # Green line
            
            # Draw a circle at the bottom point (optional)
            cv2.circle(result_img, (bottom_x, bottom_y), 5, (0, 255, 0), -1)
            
            # Store the line endpoints for later use in steering calculations
            center_line_top = (top_x, top_y)
            center_line_bottom = (bottom_x, bottom_y)
            
            # Calculate the center line angle
            angle = self.get_center_line_angle(center_line_top, center_line_bottom)
            
            # Add angle to the image (optional)
            cv2.putText(result_img, f"Angle: {angle:.1f}°", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return result_img, center_line_top, center_line_bottom, angle
            
        except Exception as e:
            print(f"Error fitting line: {e}")
            return result_img, None, None, None

    def load_gps_path(self, csv_file="../map_processing/test_points_processed.csv"):
        """
        Load GPS coordinates from CSV file.
        
        Args:
            csv_file: Path to the CSV file with GPS coordinates
            
        Returns:
            Array of GPS coordinates [(x1,y1), (x2,y2), ...]
        """
        gps_coords = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header if present
                for row in reader:
                    # Assuming the CSV has x,y coordinates in the first two columns
                    x, y = float(row[0]), float(row[1])
                    gps_coords.append((x, y))
        except Exception as e:
            print(f"Error loading GPS coordinates: {e}")
        
        return gps_coords

    def find_nearest_path_point(self, current_position, path_points):
        """
        Find the nearest point on the path and the next target point.
        
        Args:
            current_position: Tuple (x, y) of current position
            path_points: List of path points [(x1,y1), (x2,y2), ...]
            
        Returns:
            Tuple of (nearest_point_index, next_point_index)
        """
        distances = [np.sqrt((p[0]-current_position[0])**2 + (p[1]-current_position[1])**2) 
                    for p in path_points]
        
        nearest_idx = np.argmin(distances)
        next_idx = (nearest_idx + 1) % len(path_points)
        
        return nearest_idx, next_idx

    def determine_steering_command(self, center_line_angle, target_path_angle, vision_weight=0.7):
        """
        Combine vision-based and GPS-based steering commands.
        
        Args:
            center_line_angle: Angle of the center line from vision (degrees)
            target_path_angle: Angle to the next GPS target point (degrees)
            vision_weight: Weight for vision-based steering (0-1)
            
        Returns:
            Final steering angle (degrees)
        """
        # Combine the two angles with weighted average
        gps_weight = 1 - vision_weight
        final_angle = vision_weight * center_line_angle + gps_weight * target_path_angle
        
        return final_angle

    def get_center_line_angle(self, top_point, bottom_point):
        """
        Calculate the angle of the center line.
        
        Args:
            top_point: (x, y) coordinates of the top of the line
            bottom_point: (x, y) coordinates of the bottom of the line
            
        Returns:
            Angle in degrees
        """
        dx = top_point[0] - bottom_point[0]
        dy = top_point[1] - bottom_point[1]
        
        # Calculate angle in degrees (0 is vertical, positive is right, negative is left)
        angle = np.arctan2(dx, dy) * 180 / np.pi
        
        return angle

    def visualize_robot_on_map(self, robot_x, robot_y, robot_angle, center_line_angle=None, 
                              map_path="../map_processing/final_path.png", 
                              pixel_size=0.03, origin_pixel=(2418.8, 175.3)):
        """
        Visualize the robot's position and direction on the map using image_transform_reverse2.
        
        Args:
            robot_x: Real-world x coordinate of the robot (meters)
            robot_y: Real-world y coordinate of the robot (meters)
            robot_angle: Heading angle of the robot (degrees)
            center_line_angle: Optional angle from the vision center line (degrees)
            map_path: Path to the map image
            pixel_size: Size of each pixel in the map (meters)
            origin_pixel: Pixel coordinates of the origin (0,0) on the map
            
        Returns:
            Map image with robot position and direction drawn
        """
        # Load the map image
        map_img = cv2.imread(map_path)
        if map_img is None:
            raise ValueError(f"Could not read map image from {map_path}")
        
        # Create a copy of the map image to draw on for the original view
        result_img = map_img.copy()
        
        # Convert robot real-world coordinates to pixel coordinates
        robot_px, robot_py = real_coords_to_pixel_coords(robot_x, robot_y, pixel_size, origin_pixel)
        
        # Draw robot position as a circle on original map
        cv2.circle(result_img, (int(robot_px), int(robot_py)), 15, (0, 0, 255), -1)  # Red circle
        
        # Draw robot heading direction on original map
        heading_length = 50  # Length of the direction indicator
        heading_angle = radians(robot_angle + 180)  # Convert to radians and adjust
        heading_px = robot_px + heading_length * cos(heading_angle)
        heading_py = robot_py - heading_length * sin(heading_angle)
        
        cv2.line(result_img, 
                 (int(robot_px), int(robot_py)), 
                 (int(heading_px), int(heading_py)), 
                 (0, 0, 255), 3)  # Red line for robot heading
        
        # Draw vision direction line if center_line_angle is provided
        if center_line_angle is not None:
            combined_angle_rad = radians(robot_angle + center_line_angle)
            vision_px = robot_px + heading_length * 1.5 * cos(combined_angle_rad)
            vision_py = robot_py - heading_length * 1.5 * sin(combined_angle_rad)
            
            cv2.line(result_img, 
                    (int(robot_px), int(robot_py)), 
                    (int(vision_px), int(vision_py)), 
                    (0, 255, 0), 3)  # Green line for vision direction
            
        # Add labels to original map
        cv2.putText(result_img, f"Position: ({robot_x:.2f}m, {robot_y:.2f}m)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(result_img, f"Heading: {robot_angle:.1f}°", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if center_line_angle is not None:
            cv2.putText(result_img, f"Vision angle: {center_line_angle:.1f}°", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Get the warped image from robot's perspective using image_transform_reverse2
        # Using smaller output scale for better visualization
        out_scale = 0.25
        
        # Suppress debug prints from get_warped_image
        with StringIO() as buf, redirect_stdout(buf):
            warped_img = get_warped_image(
                x=robot_x, 
                y=robot_y, 
                theta=robot_angle, 
                map_image=map_img,
                out_scale=out_scale
            )
        
        # Create a combined visualization
        # Get dimensions
        map_height, map_width = result_img.shape[:2]
        warped_height, warped_width = warped_img.shape[:2]
        
        # Create a composite image
        composite_height = max(map_height, warped_height)
        composite_width = map_width + warped_width
        composite_img = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Add the map visualization to the left
        composite_img[0:map_height, 0:map_width] = result_img
        
        # Add the warped image to the right
        composite_img[0:warped_height, map_width:map_width+warped_width] = warped_img
        
        # Add labels to identify each image
        cv2.putText(composite_img, "Map View with Robot Position", 
                    (10, map_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite_img, "Robot's Perspective View", 
                    (map_width + 10, warped_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save the visualization
        output_path = "robot_on_map.png"
        cv2.imwrite(output_path, composite_img)
        print(f"Robot visualization saved to {output_path}")
        
        # Display the image
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Robot at ({robot_x}m, {robot_y}m), heading {robot_angle}°")
        plt.tight_layout()
        plt.show()
        
        return composite_img

    def visualize_center_line_on_map(self, image_path, robot_x, robot_y, robot_angle,
                                   map_path="../map_processing/final_path.png",
                                   pixel_size=0.03, origin_pixel=(2418.8, 175.3),
                                   show_path_points=3, look_ahead=20, vision_weight=0.6):
        """
        Visualize the center line from the camera image on the map alongside robot position.
        
        Args:
            image_path: Path to the processed center line image
            robot_x, robot_y: Robot position in meters
            robot_angle: Robot heading in degrees
            map_path: Path to the map image
            pixel_size: Size of each pixel in meters
            origin_pixel: Origin coordinates in pixels
            show_path_points: Number of future path points to show
            look_ahead: Number of points to look ahead for drawing a path line
            vision_weight: Weight for vision-based steering (0-1)
        """
        # Load the center line image
        center_img = cv2.imread(image_path)
        
        # Load the map image
        map_img = cv2.imread(map_path)

        # Create a copy of the map image to draw on
        result_img = map_img.copy()
        
        # Convert robot real-world coordinates to pixel coordinates
        robot_px, robot_py = real_coords_to_pixel_coords(robot_x, robot_y, pixel_size, origin_pixel)
        
        # Draw robot position as a circle
        cv2.circle(result_img, (int(robot_px), int(robot_py)), 15, (0, 0, 255), -1)  # Red circle
        
        # Draw robot heading direction
        heading_length = 50
        heading_angle = radians(robot_angle + 180)
        heading_px = robot_px + heading_length * cos(heading_angle)
        heading_py = robot_py - heading_length * sin(heading_angle)
        cv2.line(result_img, 
                 (int(robot_px), int(robot_py)), 
                 (int(heading_px), int(heading_py)), 
                 (0, 0, 255), 3)  # Red line
        
        # Variables to store angles for steering calculation
        vision_center_line_angle = None
        look_ahead_path_angle = None
        
        # Extract the center line angle from the center line image
        # The angle info is already in the image text, but let's extract it more directly
        # Look for green pixels in a line (the center line)
        center_gray = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
        _, center_mask = cv2.threshold(center_gray, 127, 255, cv2.THRESH_BINARY)
        center_bgr = cv2.cvtColor(center_mask, cv2.COLOR_GRAY2BGR)
        center_hsv = cv2.cvtColor(center_img, cv2.COLOR_BGR2HSV)

        # Green color range in HSV
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(center_hsv, lower_green, upper_green)

        # Find the green line endpoints
        green_points = np.where(green_mask > 0)
        if len(green_points[0]) > 0:
            # Find top and bottom points of the green line
            y_values = green_points[0]
            x_values = green_points[1]
            
            # Group points by y coordinate to find the center line
            y_sorted_indices = np.argsort(y_values)
            y_sorted = y_values[y_sorted_indices]
            x_sorted = x_values[y_sorted_indices]
            
            # Get the top and bottom points
            top_idx = 0
            bottom_idx = len(y_sorted) - 1
            
            # Get multiple points and average to reduce noise
            top_n = min(20, len(y_sorted) // 10)
            bottom_n = min(20, len(y_sorted) // 10)
            
            top_y = int(np.mean(y_sorted[:top_n]))
            top_x = int(np.mean(x_sorted[:top_n]))
            
            bottom_y = int(np.mean(y_sorted[-bottom_n:]))
            bottom_x = int(np.mean(x_sorted[-bottom_n:]))
            
            # Calculate center line angle from the camera view
            vision_center_line_angle = np.arctan2(top_x - bottom_x, top_y - bottom_y) * 180 / np.pi
            
            # Now project this line onto the map
            # The vision center line starts at the robot position and extends forward
            vision_line_length = 100  # Adjust as needed
            
            # Calculate the combined angle (robot heading + center line angle)
            combined_angle_rad = radians(robot_angle + vision_center_line_angle)
            
            # Start at robot position and extend along the combined angle
            vision_end_x = robot_px + vision_line_length * cos(combined_angle_rad)
            vision_end_y = robot_py - vision_line_length * sin(combined_angle_rad)
            
            # Draw the projected center line on the map
            cv2.line(result_img, 
                     (int(robot_px), int(robot_py)), 
                     (int(vision_end_x), int(vision_end_y)), 
                     (0, 255, 0), 3)  # Green line for projected center line
            
            # Label the line
            cv2.putText(result_img, f"Vision Center Line", 
                        (int(vision_end_x) + 5, int(vision_end_y) + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Load GPS path coordinates
        gps_path = self.load_gps_path()
        
        if gps_path:
            # Find nearest path point
            nearest_idx, next_idx = self.find_nearest_path_point((robot_x, robot_y), gps_path)
            
            # Draw nearest point and next few points
            for i in range(nearest_idx, min(nearest_idx + show_path_points + 1, len(gps_path))):
                point_x, point_y = gps_path[i]
                point_px, point_py = real_coords_to_pixel_coords(point_x, point_y, pixel_size, origin_pixel)
            
            # Draw a look-ahead line from robot to the point that's look_ahead steps forward
            look_ahead_idx = min(nearest_idx + look_ahead, len(gps_path) - 1)
            if look_ahead_idx > nearest_idx:
                look_ahead_x, look_ahead_y = gps_path[look_ahead_idx]
                look_ahead_px, look_ahead_py = real_coords_to_pixel_coords(look_ahead_x, look_ahead_y, pixel_size, origin_pixel)
                
                # Draw the look-ahead line
                cv2.line(result_img, 
                         (int(robot_px), int(robot_py)), 
                         (int(look_ahead_px), int(look_ahead_py)), 
                         (0, 255, 255), 3)  # Yellow line for look-ahead path
                
                # Draw the look-ahead point with a different color
                cv2.circle(result_img, (int(look_ahead_px), int(look_ahead_py)), 5, (0, 255, 255), -1)  # Yellow
                
                # Label the look-ahead point
                cv2.putText(result_img, f"Look Ahead ({look_ahead})", 
                            (int(look_ahead_px) + 10, int(look_ahead_py) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Calculate look-ahead path angle
                dx = look_ahead_x - robot_x
                dy = look_ahead_y - robot_y
                look_ahead_path_angle = np.arctan2(dx, dy) * 180 / np.pi - robot_angle
                
                # Normalize angle to be between -180 and 180
                while look_ahead_path_angle > 180:
                    look_ahead_path_angle -= 360
                while look_ahead_path_angle < -180:
                    look_ahead_path_angle += 360
        
        # Calculate and draw the steering direction if both angles are available
        if vision_center_line_angle is not None and look_ahead_path_angle is not None:
            # Calculate steering angle using determine_steering_command
            steering_angle = self.determine_steering_command(
                vision_center_line_angle, 
                look_ahead_path_angle, 
                vision_weight=vision_weight
            )
            
            # Draw the steering direction line
            steering_line_length = 120  # Longer than other lines for clarity
            steering_angle_rad = radians(robot_angle + steering_angle)
            
            steering_end_x = robot_px + steering_line_length * cos(steering_angle_rad)
            steering_end_y = robot_py - steering_line_length * sin(steering_angle_rad)
            
            # Draw the steering direction line with a magenta color
            cv2.line(result_img, 
                     (int(robot_px), int(robot_py)), 
                     (int(steering_end_x), int(steering_end_y)), 
                     (255, 0, 255), 4)  # Magenta line for steering
            
            # Add a label for the steering line
            cv2.putText(result_img, f"Steering", 
                        (int(steering_end_x) + 5, int(steering_end_y) + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Add more detailed angle information at the bottom of the image
            angle_info = f"Vision: {vision_center_line_angle:.1f}  Path: {look_ahead_path_angle:.1f}  Steering: {steering_angle:.1f}"
            map_height = result_img.shape[0]
            cv2.putText(result_img, angle_info, 
                        (10, map_height - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
        
        # Create a combined visualization
        # Resize center line image to fit in the visualization if needed
        height, width = center_img.shape[:2]
        map_height, map_width = result_img.shape[:2]
        
        # Create a composite image
        composite_height = max(map_height, height)
        composite_width = map_width + width
        composite_img = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Add the map visualization to the left
        composite_img[0:map_height, 0:map_width] = result_img
        
        # Add the center line image to the right
        composite_img[0:height, map_width:map_width+width] = center_img
        
        # Add a label to identify each image
        cv2.putText(composite_img, "Map View with Robot Position", 
                    (10, map_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(composite_img, "Camera View with Center Line", 
                    (map_width + 10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the composite image
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Robot at ({robot_x}m, {robot_y}m), heading {robot_angle}°")
        plt.tight_layout()
        plt.show()
        
        return composite_img

if __name__ == "__main__":
    control = Control()
    
    # Visualize the robot and center line together
    control.visualize_center_line_on_map(
        image_path="center_line.png",
        robot_x=36, 
        robot_y=-0.75, 
        robot_angle=-40,
        show_path_points=5,
        look_ahead=20,
        vision_weight=0.7
    )
