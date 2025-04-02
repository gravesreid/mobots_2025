import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_transform_matrix(real_coords, pixel_coords, pixel_size=0.003):
    """
    Calculate the perspective transform matrix from camera view to flat plane,
    ensuring the full visible floor area is included.
    
    Args:
        real_coords: List of (x, y) coordinates in real-world space (meters)
        pixel_coords: List of (i, j) coordinates in the image (pixels)
        pixel_size: Size of each pixel in the output image (meters/pixel)
        
    Returns:
        The perspective transform matrix, x_offset, y_offset, output_size
    """
    # Find the extreme values of the coordinates
    x_values = [x for x, y in real_coords]
    y_values = [y for x, y in real_coords]
    
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    
    # Calculate the range of known coordinates
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Add a buffer (50% of range) to ensure we capture the entire visible floor
    buffer_x = x_range * 0.5
    buffer_y = y_range * 0.5
    
    # Calculate the total visible area with buffer
    total_min_x = min_x - buffer_x
    total_max_x = max_x + buffer_x
    total_min_y = min_y - buffer_y
    total_max_y = max_y + buffer_y
    
    print(f"Estimated visible floor area: X: {total_min_x} to {total_max_x}, Y: {total_min_y} to {total_max_y}")
    
    # Calculate required offsets to make all coordinates positive
    x_offset = abs(min(0, total_min_x))
    y_offset = abs(min(0, total_min_y))
    
    print(f"Applied offsets: x_offset={x_offset}, y_offset={y_offset}")
    
    # Apply offset to coordinates
    offset_real_coords = [(x + x_offset, y + y_offset) for x, y in real_coords]
    
    print(f"Original coordinates: {real_coords}")
    print(f"Offset coordinates: {offset_real_coords}")
    
    # Calculate the output image size needed
    width = int(np.ceil((total_max_x + x_offset) / pixel_size))
    height = int(np.ceil((total_max_y + y_offset) / pixel_size))
    output_size = (width, height)
    
    print(f"Calculated output size: {output_size}")
    
    # Convert real-world coordinates to pixel coordinates
    scale = 1.0 / pixel_size
    dst_points = np.array([[x * scale, y * scale] for x, y in offset_real_coords], dtype=np.float32)
    
    # Convert image pixel coordinates to the format expected by cv2.getPerspectiveTransform
    src_points = np.array([[i, j] for i, j in pixel_coords], dtype=np.float32)
    
    print("src_points (pixel coordinates):")
    print(src_points)
    print("dst_points (scaled real-world coordinates with offset):")
    print(dst_points)
    
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return M, x_offset, y_offset, output_size

def transform_image(image_path, transform_matrix, output_size):
    """
    Apply perspective transform to an image.
    
    Args:
        image_path: Path to the input image
        transform_matrix: The perspective transform matrix
        output_size: Size of the output image (width, height) in pixels
        
    Returns:
        The transformed image
    """
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Apply the perspective transform
    transformed_img = cv2.warpPerspective(img, transform_matrix, output_size)
    
    return transformed_img

def save_transformed_image(transformed_img, output_path):
    """
    Save the transformed image to a file.
    
    Args:
        transformed_img: The transformed image
        output_path: Path to save the output image
    """
    cv2.imwrite(output_path, transformed_img)
    print(f"Transformed image saved to {output_path}")

def draw_reference_points(img, pixel_coords):
    """
    Draw the reference points on the image for visualization.
    
    Args:
        img: The image
        pixel_coords: List of (i, j) coordinates to mark
        
    Returns:
        Image with marked reference points
    """
    marked_img = img.copy()
    
    for i, (x, y) in enumerate(pixel_coords):
        cv2.circle(marked_img, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red circle
        cv2.putText(marked_img, f"P{i+1}", (int(x)+15, int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return marked_img

def draw_transformed_points(img, real_coords, x_offset, y_offset, pixel_size=0.003):
    """
    Draw the reference points on the transformed image for verification.
    
    Args:
        img: The transformed image
        real_coords: List of (x, y) coordinates in real space
        x_offset: X offset applied to handle negative coordinates
        y_offset: Y offset applied to handle negative coordinates
        pixel_size: Size of each pixel in the output image (meters/pixel)
        
    Returns:
        Image with marked reference points
    """
    marked_img = img.copy()
    scale = 1.0 / pixel_size
    
    for i, (x, y) in enumerate(real_coords):
        # Apply offset and scale coordinates
        px = int((x + x_offset) * scale)
        py = int((y + y_offset) * scale)
        
        cv2.circle(marked_img, (px, py), 10, (0, 255, 0), -1)  # Green circle
        cv2.putText(marked_img, f"P{i+1}", (px+15, py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return marked_img

def draw_grid(img, x_offset, y_offset, grid_size=0.1, pixel_size=0.003, line_count=20):
    """
    Draw a grid on the transformed image to help with measurement.
    
    Args:
        img: The transformed image
        x_offset: X offset applied to handle negative coordinates
        y_offset: Y offset applied to handle negative coordinates
        grid_size: Size of grid cells in meters
        pixel_size: Size of each pixel in the output image (meters/pixel)
        line_count: Number of grid lines to draw in each direction
        
    Returns:
        Image with grid
    """
    grid_img = img.copy()
    h, w = grid_img.shape[:2]
    scale = 1.0 / pixel_size
    
    # Draw vertical lines (constant x)
    for i in range(-line_count, line_count + 1):
        # Calculate grid position in meters (original coordinates)
        x_pos_meters = i * grid_size
        # Convert to pixel position (with offset applied)
        x_pos_pixels = int((x_pos_meters + x_offset) * scale)
        
        if 0 <= x_pos_pixels < w:
            cv2.line(grid_img, (x_pos_pixels, 0), (x_pos_pixels, h), (0, 255, 255), 1)
            # Label the grid line
            cv2.putText(grid_img, f"{x_pos_meters:.1f}m", (x_pos_pixels + 5, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(grid_img, f"{x_pos_meters:.1f}m", (x_pos_pixels + 5, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw horizontal lines (constant y)
    for i in range(-line_count, line_count + 1):
        # Calculate grid position in meters (original coordinates)
        y_pos_meters = i * grid_size
        # Convert to pixel position (with offset applied)
        y_pos_pixels = int((y_pos_meters + y_offset) * scale)
        
        if 0 <= y_pos_pixels < h:
            cv2.line(grid_img, (0, y_pos_pixels), (w, y_pos_pixels), (0, 255, 255), 1)
            # Label the grid line
            cv2.putText(grid_img, f"{y_pos_meters:.1f}m", (5, y_pos_pixels + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(grid_img, f"{y_pos_meters:.1f}m", (5, y_pos_pixels + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return grid_img

def plot_comparison(original_img, transformed_img):
    """
    Plot the original and transformed images side by side.
    
    Args:
        original_img: The original image
        transformed_img: The transformed image
    """
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Camera View")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    plt.title("Transformed Top-Down View")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Given data
    pixel_coords = [(1142, 629), (245, 239), (1025, 145), (1878, 276)]
    real_coords = [(1, 0), (2, 1), (3, 0), (2, -1)]
    
    # real_coords = [(-y, -x) for x, y in real_coords]
    pixel_size = 0.003  # 3mm per pixel
    
    # Calculate the transform matrix with offset handling and appropriate output size
    transform_matrix, x_offset, y_offset, output_size = calculate_transform_matrix(
        real_coords, pixel_coords, pixel_size)

    print("Transform Matrix:")
    print(transform_matrix)
    
    # Apply the transform to an image
    image_path = "/home/aigeorge/projects/mobots_2025/data/image_20250401_184515_960.jpg"  # Replace with your image path
    
    try:
        # Read original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Mark reference points on original image
        marked_original = draw_reference_points(original_img, pixel_coords)
        
        # Transform the image
        transformed_img = transform_image(image_path, transform_matrix, output_size)
        
        # Mark reference points on transformed image
        marked_transformed = draw_transformed_points(transformed_img, real_coords, x_offset, y_offset, pixel_size)
        
        # Add grid to transformed image
        grid_transformed = draw_grid(marked_transformed, x_offset, y_offset, 0.5, pixel_size, 20)
        
        # Save the transformed images
        save_transformed_image(transformed_img, "transformed_image.jpg")
        save_transformed_image(marked_original, "marked_original.jpg")
        save_transformed_image(marked_transformed, "marked_transformed.jpg")
        save_transformed_image(grid_transformed, "grid_transformed.jpg")
        
        # Plot comparison
        plot_comparison(marked_original, grid_transformed)
        
        print("Transformation complete!")
        print(f"Applied offsets: x_offset={x_offset}, y_offset={y_offset}")
        print(f"Output image size: {output_size}")
        
    except Exception as e:
        print(f"Error: {e}")