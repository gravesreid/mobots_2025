import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sin, cos, radians

def real_coords_to_pixel_coords(x, y, pixel_size=0.03, origin_pixel=(2418.8, 175.3)) -> tuple[float, float]:
    """
    Convert real-world coordinates to pixel coordinates on the map.
    
    Args:
        real_coords: List of (x, y) coordinates in real-world space (meters)
        pixel_size: Size of each pixel in the map (meters)
        origin_pixel: Pixel coordinates of the origin (0,0) on the map

    Returns:
        The pixel coordinates on the map
    """
    # Convert from real-world to pixel coordinates
    map_x = origin_pixel[0] - x / pixel_size
    map_y = origin_pixel[1] + y / pixel_size

    return map_x, map_y

def calculate_camera_transform(real_coords, pixel_coords, pixel_size=0.03):
    """
    Calculate the transform matrix to go from map to camera view.
    
    Args:
        real_coords: List of (x, y) coordinates in real-world space (meters)
        pixel_coords: List of (i, j) coordinates in the camera image (pixels)
        pixel_size: Size of each pixel in the map (meters)
        
    Returns:
        The perspective transform matrix
    """
    # Convert real-world coordinates to map pixel coordinates
    origin_pixel = (2418.8, 175.3)
    
    map_coords = []
    for x, y in real_coords:
        # Convert from real-world to pixel coordinates
        map_x, map_y = real_coords_to_pixel_coords(x, y, pixel_size, origin_pixel)
        map_coords.append((map_x, map_y))
    
    # For the transform, we want to go from the map to the camera view
    src_points = np.array(map_coords, dtype=np.float32)
    dst_points = np.array(pixel_coords, dtype=np.float32)
    
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return M

def simulate_camera_view(map_path, camera_x, camera_y, camera_theta, 
                         transform_matrix, origin_pixel=(2418.8, 175.3),
                         pixel_size=0.03, output_size=(1920, 1080)):
    """
    Simulate camera view from specified position and orientation.
    
    Args:
        map_path: Path to the map image
        camera_x: X-coordinate of camera in real-world space (meters)
        camera_y: Y-coordinate of camera in real-world space (meters)
        camera_theta: Orientation of camera in degrees (0 = facing along positive Y-axis)
        transform_matrix: The perspective transform matrix from reference position
        origin_pixel: Real-world coordinates of the reference point for the transform
        pixel_size: Size of each pixel in the map (meters)
        output_size: Size of the output camera image (width, height)
        
    Returns:
        The simulated camera view and the map with marked camera position
    """
    # Read the map image
    map_img = cv2.imread(map_path)
    if map_img is None:
        raise ValueError(f"Could not read map image from {map_path}")
    
    # The transform matrix was calculated for a specific reference position/orientation
    # We need to adjust the map to account for the new position and orientation

    map_x, map_y = real_coords_to_pixel_coords(camera_x, camera_y, pixel_size, reference_point)
    
    # 1. Calculate the translation needed (difference between reference and new position)
    dx = map_x - origin_pixel[0]
    dy = map_y - origin_pixel[1]
    
    # 3. Create a transformation matrix to translate and rotate the map
    # Note: OpenCV uses column-major ordering
    # The rotation is around the camera position
    
    # Create translation matrix
    translation_matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    # adjust for the definition of the origin:
    
    # Create rotation matrix
    # rotation_matrix = cv2.getRotationMatrix2D((map_x, map_y), -camera_theta, 1.0)
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), -camera_theta, 1.0)
    
    # Apply translation and rotation to the map
    map_height, map_width = map_img.shape[:2]
    
    # First translate
    translated_map = cv2.warpAffine(map_img, translation_matrix, (map_width, map_height))
    
    # Then rotate 
    rotated_map = cv2.warpAffine(translated_map, rotation_matrix, (map_width, map_height))
    
    # Now apply the perspective transform to simulate camera view
    camera_view = cv2.warpPerspective(
        rotated_map, transform_matrix, output_size, 
        flags=cv2.INTER_LINEAR
    )
    
    # Mark camera position on the original map
    marked_map = draw_camera_position(map_img.copy(), camera_x, camera_y, camera_theta, 
                                      origin_pixel, pixel_size)
    
    return marked_map, camera_view

def draw_camera_position(map_img, camera_x, camera_y, camera_theta, 
                         origin_pixel=(2418.8, 175.3), pixel_size=0.03):
    """
    Draw camera position and orientation on the map.
    
    Args:
        map_img: Map image
        camera_x: X-coordinate of camera in real-world space (meters)
        camera_y: Y-coordinate of camera in real-world space (meters)
        camera_theta: Orientation of camera in degrees (0 = facing positive Y direction)
        origin_pixel: Pixel coordinates of the origin (0,0) on the map
        pixel_size: Size of each pixel in the map (meters)
        
    Returns:
        Map image with camera position marked
    """
    # Convert camera position from real-world coordinates to pixel coordinates
    camera_pixel_x = int(origin_pixel[0] - camera_x / pixel_size)
    camera_pixel_y = int(origin_pixel[1] + camera_y / pixel_size)
    
    # Calculate end point of direction indicator
    length = 50  # pixels
    theta_rad = radians(camera_theta)
    direction_x = int(camera_pixel_x + length * sin(theta_rad))
    direction_y = int(camera_pixel_y - length * cos(theta_rad))
    
    # Draw camera position
    cv2.circle(map_img, (camera_pixel_x, camera_pixel_y), 10, (0, 0, 255), -1)
    
    # Draw camera direction
    cv2.line(map_img, (camera_pixel_x, camera_pixel_y), (direction_x, direction_y), (0, 0, 255), 2)
    
    # Add text to show coordinates and angle
    cv2.putText(map_img, f"({camera_x}, {camera_y}), θ={camera_theta}°", 
                (camera_pixel_x + 15, camera_pixel_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return map_img

def draw_map_points(map_img, real_coords, pixel_size=0.03, origin_pixel=(2418.8, 175.3)):
    """
    Draw the reference points on the map for visualization.
    
    Args:
        map_img: The map image
        real_coords: List of (x, y) coordinates in real space
        pixel_size: Size of each pixel in the map (meters)
        origin_pixel: Pixel coordinates of the origin (0,0) on the map
        
    Returns:
        Map image with marked reference points
    """
    marked_img = map_img.copy()
    
    for i, (x, y) in enumerate(real_coords):
        # Convert from real-world to pixel coordinates
        map_x = int(origin_pixel[0] - x / pixel_size)
        map_y = int(origin_pixel[1] + y / pixel_size)
        
        cv2.circle(marked_img, (map_x, map_y), 10, (0, 255, 0), -1)  # Green circle
        cv2.putText(marked_img, f"P{i+1}", (map_x+15, map_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return marked_img

def draw_camera_view_points(camera_view, pixel_coords):
    """
    Draw the reference points on the camera view for verification.
    
    Args:
        camera_view: The camera view image
        pixel_coords: List of (i, j) coordinates in the camera image
        
    Returns:
        Camera view with marked reference points
    """
    marked_img = camera_view.copy()
    
    for i, (x, y) in enumerate(pixel_coords):
        cv2.circle(marked_img, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red circle
        cv2.putText(marked_img, f"P{i+1}", (int(x)+15, int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return marked_img

def plot_comparison(map_img, camera_view, camera_x, camera_y, camera_theta):
    """
    Plot the map and simulated camera view side by side.
    
    Args:
        map_img: The map image
        camera_view: The simulated camera view
        camera_x: X-coordinate of camera
        camera_y: Y-coordinate of camera
        camera_theta: Orientation of camera in degrees
    """
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
    plt.title("Map View with Camera Position")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB))
    plt.title(f"Simulated Camera View at ({camera_x}, {camera_y}), θ={camera_theta}°")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_image(img, output_path):
    """
    Save an image to a file.
    
    Args:
        img: The image to save
        output_path: Path to save the image
    """
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Define the reference points
    pixel_coords = [(1142, 629), (245, 239), (1025, 145), (1878, 276)]
    real_coords = [(1, 0), (2, 1), (3, 0), (2, -1)]
    
    # Convert to the correct coordinate system
    # real_coords = [(-y, -x) for x, y in real_coords]
    
    # Calculate the base transform matrix from the reference position
    # This is the transform for a camera at position (0, 0) with orientation 0°
    transform_matrix = calculate_camera_transform(real_coords, pixel_coords)
    
    # Set the reference point (where the original transform was calculated)
    # This should be one of the points used to calculate the transform
    reference_point = real_coords[0]  # Using first point as reference
    
    print(f"Using reference point: {reference_point}")
    print("Transform Matrix:")
    print(transform_matrix)
    
    # Path to the map image
    map_path = "/home/aigeorge/projects/mobots_2025/map_processing/Mobot Satalite - crop.png"
    output_size = (1920, 1080)  # Output camera view size
    
    # Define camera positions to simulate
    camera_positions = [
        # (x, y, theta)
        (5, -1, 0),      # At origin, facing "north"
        (10, -2, 90),     # At origin, facing "east"
        (10, 0, 0),      # 1m east of origin, facing "north"
        (10, -1, 180),    # 1m north of origin, facing "south"
        (15, -1, 45)    # 1m southwest of origin, facing northeast
    ]
    
    try:
        for i, (camera_x, camera_y, camera_theta) in enumerate(camera_positions):
            # Simulate camera view from this position
            marked_map, camera_view = simulate_camera_view(
                map_path, camera_x, camera_y, camera_theta,
                transform_matrix, reference_point
            )
            
            # Save images
            save_image(camera_view, f"camera_view_pos{i}.jpg")
            save_image(marked_map, f"map_with_camera_pos{i}.jpg")
            
            # Plot comparison
            plot_comparison(marked_map, camera_view, camera_x, camera_y, camera_theta)
            
            print(f"Simulation complete for camera at ({camera_x}, {camera_y}), θ={camera_theta}°")
        
    except Exception as e:
        print(f"Error: {e}")