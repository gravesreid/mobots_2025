import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sin, cos, radians
from typing import List, Tuple

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


def get_warped_image(x: float, 
                     y:float, 
                     theta:float, 
                     map_image:np.ndarray, 
                     out_scale:float = 1.0,
                     image_size:Tuple[int, int] = (1920, 1080),
                     calibration_pix: List[Tuple[float, float]] = [(1142, 629), (245, 239), (1025, 145), (1878, 276)],
                     calibration_loc: List[Tuple[float, float]] = [(1, 0), (2, 1), (3, 0), (2, -1)]
                     ) -> np.ndarray:
    """
    Get the warped image from the camera view.
    
    Args:
        x: The x position of the camera in the map
        y: The y position of the camera in the map
        theta: The angle of the camera in the map (degrees)
        src_image: The map image
        out_scale: The scale of the output image (default 1.0) 
        image_size: The size of calibration image
        calibration_pix: The pixel coordinates of the calibration points
        calibration_loc: The location of the calibration points in the map
        
    Returns:
        The warped image from the camera view
    """
    # convert the real-world coordinates to pixel coordinates

    if out_scale != 1.0:
        image_size = (int(image_size[0] * out_scale), int(image_size[1] * out_scale))
        calibration_pix = [(pt[0] * out_scale, pt[1] * out_scale) for pt in calibration_pix]
        
    
    new_calib_points = []
    for src_point in calibration_loc:
        # given the x and y displacment, and the angle of the camera, calculate the new calibration poitns (world coordinates)
        x_shift = x + src_point[0] * cos(radians(theta)) - src_point[1] * sin(radians(theta))
        y_shift = y + src_point[0] * sin(radians(theta)) + src_point[1] * cos(radians(theta))

        # convert the real-world coordinates to pixel coordinates
        new_calib_points.append(real_coords_to_pixel_coords(x_shift, y_shift))

    # print("new_calib_points: ", new_calib_points)
    # print("calibration_pix: ", calibration_pix)

    calibration_pix = np.array(calibration_pix, dtype=np.float32)
    new_calib_points = np.array(new_calib_points, dtype=np.float32)

    # calculate the perspective transform matrix
    # M = cv2.getPerspectiveTransform(calibration_pix, new_calib_points)
    M = cv2.getPerspectiveTransform(new_calib_points, calibration_pix)

    # print(M)

    # Warp the image
    dst_image = cv2.warpPerspective(map_image, M, image_size)

    return dst_image


# Test the functions
if __name__ == "__main__":
    # Load the source image
    src_image = cv2.imread("map_processing/final_path.png")

    # Define the camera position and angle
    x = 10
    y = 0
    theta = 0

    # Get the warped image
    dst_image = get_warped_image(x, y, theta, src_image, out_scale=0.25)

    x_pix, y_pix = real_coords_to_pixel_coords(x, y)

    # Display the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Source Image")
    plt.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    plt.plot(x_pix, y_pix, "ro")  # Camera position

    # add a line for the camera angle
    angle = radians(theta + 180)
    x2 = x_pix + 100 * cos(angle)
    y2 = y_pix - 100 * sin(angle)

    plt.plot([x_pix, x2], [y_pix, y2], "r-")  # Camera angle
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Warped Image")
    plt.imshow(cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

