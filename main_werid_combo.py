import lgpio
import time
import numpy as np
from typing import Tuple, Union, Dict, List
import os
from headless_visualization import SimpleStreamServer
import cv2
from datetime import datetime
import threading

class MapRenderer():
    def __init__(self, map_path:str, line_step:int = 10):
        self.base_map = cv2.imread(map_path)
        self.path_idx = 0
        self.line_step = line_step

    def plot_path(self, path:np.ndarray, angle:float = 0) -> np.ndarray:
        # update the line from the last point to the current point
        if len(path) < 2:
            return self.base_map
        for i in range(self.path_idx, len(path) - 1, self.line_step):
            self.path_idx = i
            cv2.line(self.base_map, tuple(path[i].astype(np.int64)), tuple(path[i+1].astype(np.int64)), (0, 0, 255), 4)

        
        map_image = self.base_map.copy()
        # add the last point
        cv2.circle(map_image, tuple(path[-1].astype(np.int64)), 10, (255, 0, 0), -1)

        # add a line to show the heading
        x = path[-1][0]
        y = path[-1][1]
        x2 = x - 20*np.cos(angle)
        y2 = y + 20*np.sin(angle)

        cv2.line(map_image, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 4)

        return map_image


lock = threading.Lock()

class MoBot():
    ENA1, IN1_A, IN1_B = 13, 6, 5  # Motor 1
    ENCODER1 = 16
    ENA2, IN2_A, IN2_B = 12, 24, 23  # Motor 2
    ENCODER2 = 17
    TICK_DIST = 0.009
    WIDTH = 0.23
    PWM_PERIOD = 0.01

    MOTOR1_GAIN = 1
    MOTOR2_GAIN = -1

    PWM_LIM = [-0.5, 0.5]

    goal_x = 0
    goal_y = 0
    goal_theta = 0

    def __init__(self, chip, verbose = False, edge_type="both"):
        self.encoder1_count = 0
        self.encoder2_count = 0

        self.x = 0
        self.y = 0
        self.theta = 0

        self.chip = chip
        self.verbose = verbose

        self.stoped = False

        # set up the encoders:
        # Claim alerts on the GPIO pins
        for gpio in [self.ENCODER1, self.ENCODER2]:
            lgpio.gpio_claim_input(self.chip, gpio)
            if edge_type == "both":
                lgpio.gpio_claim_alert(self.chip, gpio, lgpio.BOTH_EDGES)
                lgpio.callback(chip, gpio, lgpio.BOTH_EDGES, self.callback)
            elif edge_type == "rising":
                lgpio.gpio_claim_alert(self.chip, gpio, lgpio.RISING_EDGE)
                lgpio.callback(chip, gpio, lgpio.RISING_EDGE, self.callback)
            else:
                raise ValueError(f"edge_type {edge_type} not supported")
            
        self._integral_theta = 0.0
        self._last_theta_error = 0.0

        # self._integral_dist = 0.0
        # self._last_dist_error = 0.0

        self._integral_speed = 0.0
        self._last_tot_dist = 0.0

        self._integral_dist = 0.0

        self._last_time = time.time()

        self.path_idx = 0
        self.path = np.array([[0, 0]])

        self.xs = []
        self.ys = []
        self.thetas = []
        self.speed = 0

    
    def callback(self, chip, gpio, level, timestamp):
        with lock:
            if gpio == self.ENCODER1:
                self.encoder1_count += 1
                theta_new = self.theta + self.TICK_DIST/(self.WIDTH)
                self.x += 0.5*self.WIDTH*(np.sin(theta_new) - np.sin(self.theta))
                self.y += 0.5*self.WIDTH*(-np.cos(theta_new) + np.cos(self.theta))
            elif gpio == self.ENCODER2:
                self.encoder2_count += 1
                theta_new = self.theta - self.TICK_DIST/(self.WIDTH)
                self.x += 0.5*self.WIDTH*(-np.sin(theta_new) + np.sin(self.theta))
                self.y += 0.5*self.WIDTH*(np.cos(theta_new) - np.cos(self.theta))
            else:
                assert NotImplementedError, f"gpio pin {gpio} not recognized"
        
       
        self.theta = theta_new

        self.xs.append(self.x)
        self.ys.append(self.y)
        self.thetas.append(self.theta)

        if self.verbose:
            print('Encoder 1: ', self.encoder1_count)
            print('Encoder 2: ', self.encoder2_count)
            print('x:', self.x)
            print('y:', self.y)
            print('theta:', self.theta)
            print()

        # self.closed_loop_control()
        self.follow_path_control()


    def set_motor_pwms(self, pwms):     
        pwms = np.array(pwms)

        avg = (pwms[0] + pwms[1]) / 2

        # key is to keep the average speed the same, and adjust the difference

        if avg < self.PWM_LIM[0]:
            pwms[0] = self.PWM_LIM[0]
            pwms[1] = self.PWM_LIM[0]
        elif avg > self.PWM_LIM[1]:
            pwms[0] = self.PWM_LIM[1]
            pwms[1] = self.PWM_LIM[1]
        else:
            # adjust the pwm values to be within the limits, while keeping the average the same
            if pwms[0] < self.PWM_LIM[0]:
                diff = self.PWM_LIM[0] - pwms[0]
                pwms[0] += diff
                pwms[1] -= diff
            elif pwms[0] > self.PWM_LIM[1]:
                diff = pwms[0] - self.PWM_LIM[1]
                pwms[0] -= diff
                pwms[1] += diff
            if pwms[1] < self.PWM_LIM[0]:
                diff = self.PWM_LIM[0] - pwms[1]
                pwms[1] += diff
                pwms[0] -= diff
            elif pwms[1] > self.PWM_LIM[1]:
                diff = pwms[1] - self.PWM_LIM[1]
                pwms[1] -= diff
                pwms[0] += diff

            # assert np.mean(pwms) == avg, f"average pwm {np.mean(pwms)} not equal to {avg}"
            # assert np.all(pwms >= self.PWM_LIM[0]), f"pwm {pwms} not within limits {self.PWM_LIM[0]}"
            # assert np.all(pwms <= self.PWM_LIM[1]), f"pwm {pwms} not within limits {self.PWM_LIM[1]}"

        pwm1 = pwms[0]*self.MOTOR1_GAIN
        pwm2 = pwms[1]*self.MOTOR2_GAIN

        self._set_motor(self.ENA1, self.IN1_A, self.IN1_B, pwm1)
        self._set_motor(self.ENA2, self.IN2_A, self.IN2_B, pwm2)


    def _set_motor(self, ena, in_a, in_b, pwm_val):
        duty_cycle = abs(pwm_val) * 100.0
        freq = 1.0 / self.PWM_PERIOD

        if pwm_val >= 0:
            lgpio.gpio_write(self.chip, in_a, 1)
            lgpio.gpio_write(self.chip, in_b, 0)
        else:
            lgpio.gpio_write(self.chip, in_a, 0)
            lgpio.gpio_write(self.chip, in_b, 1)

        lgpio.tx_pwm(self.chip, ena, freq, duty_cycle)

    def stop(self):
        # lgpio.tx_pwm(self.chip, self.ENA1, 0, 0)
        # lgpio.tx_pwm(self.chip, self.ENA2, 0, 0)

        # set all directions to 0
        lgpio.gpio_write(self.chip, self.IN1_A, 0)
        lgpio.gpio_write(self.chip, self.IN1_B, 0)
        lgpio.gpio_write(self.chip, self.IN2_A, 0)
        lgpio.gpio_write(self.chip, self.IN2_B, 0)

        self.stopped = True
    
    def __del__(self):
        if not self.stoped:
            self.stop()

    def set_goal(self, x:float, y:float, theta:float):
        self.goal_x = x
        self.goal_y = y
        self.goal_theta = theta
        self.closed_loop_control()
        self.follow_path_control()

    def closed_loop_control(self):
        # PID coefficients
        Kp_theta = 0.1
        Ki_theta = 0
        Kd_theta = 0.01

        Kp_dist = 0.25
        Ki_dist = 0
        Kd_dist = 0

        goal = np.array([self.goal_x, self.goal_y])
        curr = np.array([self.x, self.y])
        dist_error = np.linalg.norm(goal - curr)

        # Initialize persistent state if not already done

        # Timing for derivative/integral
        current_time = time.time()
        dt = current_time - self._last_time if current_time != self._last_time else 1e-6
        dt = max(dt, 0.25) # max time step of 0.25s. If its longer, there is probably a problem

        # Compute heading error
        desired_theta = np.arctan2(goal[1] - curr[1], goal[0] - curr[0])
        theta_error = desired_theta - self.theta
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error)) * min(dist_error, 1.0)

        # PID terms for heading
        self._integral_theta += theta_error * dt
        derivative_theta = (theta_error - self._last_theta_error) / dt

        # PID terms for distance
        self._integral_dist += dist_error * dt
        derivative_dist = (dist_error - self._last_dist_error) / dt

        # Compute PWM signals
        theta_term = (
            Kp_theta * theta_error +
            Ki_theta * self._integral_theta +
            Kd_theta * derivative_theta
        )
        dist_term = (
            Kp_dist * dist_error +
            Ki_dist * self._integral_dist +
            Kd_dist * derivative_dist
        )

        pwm1 = theta_term + dist_term
        pwm2 = -theta_term + dist_term

        # Update memory
        self._last_theta_error = theta_error
        self._last_dist_error = dist_error
        self._last_time = current_time

        self.set_motor_pwms((pwm1, pwm2))

    def set_path(self, path:np.ndarray):
        self.path_idx = 0
        self.path = path
        self.last_time = time.time()
        self.follow_path_control()


    def follow_path_control(self):
        if self.path_idx >= len(self.path) - 2:
            print('reached end of path')
            self.set_motor_pwms((0, 0))
            return
        GOAL_SPEED = 1 # m/s
        LOOK_AHEAD = 2 # m
        DEVIATION_THRESH = 0.5 # m - the relative importance of moving back to the line. W_LOOK = min(distance_from_line / DEVIATION_THRESH, 1)

        KP_ANGLE = 0.25
        KI_ANGLE = 0
        KD_ANGLE = 0.01

        KP_SPEED = 0.5
        KI_SPEED = 0

        KI_DIST = 0.02

        dt = time.time() - self.last_time
        self.last_time = time.time()
        dt = min(dt, 0.25) # max time step of 0.25s. If its longer, there is probably a problem
        pos = np.array([self.x, self.y])

        avg_encoder = (self.encoder1_count + self.encoder2_count) / 2
        total_distance = avg_encoder * self.TICK_DIST

        cur_speed = np.clip((total_distance - self._last_tot_dist) / dt, 0, 10)

        w_speed = 0.2
        self.speed = w_speed*cur_speed + (1 - w_speed) * self.speed

        if self.verbose:
            print('speed:', self.speed)
        self._last_tot_dist = total_distance

        speed_error = GOAL_SPEED - self.speed
        self._integral_speed += speed_error * dt

        self._integral_speed = np.clip(self._integral_speed, 0, 20)


        # only move forward in the path. Check to see if we are closer to the next point
        # if we are, move to the next point
        while True:
            if self.path_idx >= len(self.path) - 3:
                break
            cur_dist = np.linalg.norm(self.path[self.path_idx] - pos)
            next_dist = np.linalg.norm(self.path[self.path_idx + 1] - pos)

            if self.path[self.path_idx][0] < pos[0]:
                # we always move forward (+x direction)
                self.path_idx += 1
                continue

            if next_dist < cur_dist:
                self.path_idx += 1
            else:  
                break

        if self.verbose:
            print('path idx:', self.path_idx)

        # get the path heading:
        tangent = self.path[self.path_idx + 1] - self.path[self.path_idx - 1]
        path_heading = np.arctan2(tangent[1], tangent[0])

        # calculate the distance from the path (perpendicular distance)
        # the distance is the cross product of the vector from the current point to the robot
        # and the tangent vector of the path (unit vector)

        dist = np.cross(pos - self.path[self.path_idx], tangent / np.linalg.norm(tangent))
        

        self._integral_dist += dist * dt
        self._integral_dist = np.clip(self._integral_dist, -10, 10)
        if self.verbose:
            print('dist:', dist)

        # find the point on the path that is LOOK_AHEAD away
        look_dist = 0
        look_idx = self.path_idx
        while look_dist < LOOK_AHEAD:
            look_idx += 1
            if look_idx >= len(self.path) - 1:
                break
            look_dist += np.linalg.norm(self.path[look_idx] - self.path[look_idx - 1])
        
        # get look_heading:
        look_vec = self.path[look_idx] - pos
        look_heading = np.arctan2(look_vec[1], look_vec[0])

        w_look = min(np.abs(dist) / DEVIATION_THRESH, 1)
        goal_heading = w_look * look_heading + (1 - w_look) * path_heading

        # add dist integral term
        angle_error = goal_heading - self.theta + KI_DIST * self._integral_dist

        # print('angle_error:', angle_error)

        # wrap the angle error
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        angle_error_dot = (angle_error - self._last_theta_error) / dt
        self._integral_theta += angle_error * dt
        self._integral_theta = np.clip(self._integral_theta, -2, 2)

        pwm_angle = KP_ANGLE * angle_error + KI_ANGLE * self._integral_theta + KD_ANGLE * angle_error_dot
        pwm_speed = KP_SPEED * speed_error + KI_SPEED * self._integral_speed

        # pwm_speed = max(pwm_speed, 0.1) # make sure the robot is always moving
        if self.verbose:
            print('pwm_speed:', pwm_speed)
            print('pwm_angle:', pwm_angle)
            print('speed_error:', speed_error)
            print('angle_error:', angle_error)
            print('integral_theta:', self._integral_theta)
            print('integral_speed:', self._integral_speed)
            print('integral_dist:', self._integral_dist)

        pwm_1 = pwm_speed + pwm_angle
        pwm_2 = pwm_speed - pwm_angle

        self.set_motor_pwms((pwm_1, pwm_2))


# Given calibration data
CALIBRATION_PIXELS = [(1142, 629), (245, 239), (1025, 145), (1878, 276)]
CALIBRATION_LOCS = [(0.34, 0), (0.68, 0.34), (1.02, 0), (0.68, -0.34)]
CALIB_IMAGE_SIZE = (1920, 1080)


"""
Computes the affine transform matrix from pixel coordinates to world coordinates
using the calibration points.

Returns:
    M: 2x3 affine transformation matrix
"""
# Convert the calibration points to numpy arrays
pixels = np.array(CALIBRATION_PIXELS, dtype=np.float32)
world = np.array(CALIBRATION_LOCS, dtype=np.float32)

# Compute the affine transformation matrix
# This solves for the matrix that best maps pixels to world coordinates
TRANSFORM = cv2.estimateAffine2D(pixels, world)[0]


def pixel_to_world(pixel_coords, transform_matrix):
    """
    Converts pixel coordinates to world coordinates using the affine transform.
    
    Args:
        pixel_coords: Tuple or list (x, y) of pixel coordinates
        transform_matrix: 2x3 affine transformation matrix
    
    Returns:
        Tuple (x, y) of world coordinates
    """
    # Convert to homogeneous coordinates (add a 1)
    pixel = np.array([[pixel_coords[0]], [pixel_coords[1]], [1]], dtype=np.float32)
    
    # Apply the transform
    world = transform_matrix @ pixel
    
    return (float(world[0]), float(world[1]))

class CenterLineDetector():

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
        bottom_third_y = height * 1 // 3
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
            # if start_at_bottom_third:
            #     top_y = bottom_third_y
            # else:
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
            
            # # Add angle to the image (optional)
            # cv2.putText(result_img, f"Angle: {angle:.1f}°", 
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return result_img, center_line_top, center_line_bottom, angle
            
        except Exception as e:
            print(f"Error fitting line: {e}")
            return result_img, None, None, None
    
    def process_image(self, img):
        """
        Process an image and visualize the robot on the map.
        
        Args:
            img: camera image
            robot_x: Current robot x position (meters)
            robot_y: Current robot y position (meters)
            robot_angle: Current robot heading angle (degrees)
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Apply median blur
        blurred = cv2.medianBlur(gray, 7)
        
        # Apply different levels of blurring
        very_blurred = cv2.medianBlur(blurred, 21)
        very_very_blurred = cv2.medianBlur(blurred, 251)
        
        # Take minimum of blurred images
        combo = cv2.min(very_blurred, very_very_blurred)
        very_blurred = combo
        
        # Compute normalized difference
        diff = np.float32(blurred)/np.float32(very_blurred)
        diff = np.uint8(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))
        
        # Apply Otsu thresholding
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological closing
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep largest contiguous area
        # Find all contiguous regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        
        # Find the largest component by area (excluding background at index 0)
        largest_mask = np.zeros_like(closed_mask)
        
        if num_labels > 1:
            largest_label = 1
            largest_area = stats[1, cv2.CC_STAT_AREA]
            
            for i in range(2, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = i
            
            # Create a new mask containing only the largest component
            largest_mask[labels == largest_label] = 255
        
        # ---------- Prepare visualization of original and mask only ----------
        # Convert mask to BGR for display
    
        
        # Draw outline curves
        outlined_image, left_curve, right_curve = self.draw_outline_curves(largest_mask, img)
        
        # Create and draw center line
        center_line_image, center_line_top, center_line_bottom, center_line_angle = self.create_center_line(left_curve, right_curve, outlined_image)
        
        return np.array((center_line_bottom, center_line_top)), center_line_image

            

import matplotlib.pyplot as plt
from optimze import MobotLocator
from scipy.interpolate import CubicSpline
from image_thresh import thresh_image

MULT = []
for i in range(480):
    for j in range(270):
        MULT.append([i, j])

MULT = np.array(MULT)

def test_simple_path():
    chip = lgpio.gpiochip_open(4)
    # load path from csv "simple_path.csv", in the form x,y,t
    base_path = np.loadtxt("map_processing/race_points.csv", delimiter=",", skiprows=1)

    base_path = base_path*np.array([1, 0.65])

    save_dir = "data/run_images"
    n = 0
    # find a non-existing directory
    while os.path.exists(os.path.join(save_dir, f"run_{n}")):
        n += 1

    save_dir = os.path.join(save_dir, f"run_{n}")
    os.makedirs(save_dir)
    print(f"Images will be saved to {save_dir}")
    server = SimpleStreamServer(port=8080)
    
    cap = cv2.VideoCapture(4)

    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            assert False, "Error: Could not read frame"
            break

    input("press enter to start")
    mobot = MoBot(chip=chip, verbose=False)
    
    mobot.set_path(base_path)   
    time.sleep(0.5)

    map_renderer = MapRenderer("/home/pi/mobots_2025/map_processing/final_path.png")
    locator = MobotLocator(max_detlas=np.array([0.5, 0.5]), step_size=np.array([0.1, 0.1]), dist_penalty=0.2, debug_print=False)
    detector = CenterLineDetector()
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
                assert False, "Error: Could not read frame"

            image_pose = np.array([mobot.x, mobot.y, mobot.theta*180/np.pi])
            mobot_path = np.array([mobot.xs, mobot.ys]).copy().T

            frame_small = cv2.resize(frame, (480, 270))

            line_points, line_image = detector.process_image(frame_small) # returns the line points in the image
            mask = thresh_image(frame_small)
            mask = mask.astype(np.float64)
            avg_pix = MULT*mask/np.sum(mask)

            avg_pix = avg_pix*CALIB_IMAGE_SIZE[0]/480.0

            vec_point = pixel_to_world((avg_pix[0], avg_pix[1]), TRANSFORM)

            # rotate and translate the point to the mobot's position
            theta = image_pose[2]*np.pi/180
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            rotated_vec = vec_point@rotation_matrix.T
            goal_pos = rotated_vec + image_pose[:2]
            mobot.set_goal(goal_pos[0], goal_pos[1])
            # two points [[x1, y1], [x2, y2]] - start and end of the line. In CV2 coordinate frame
            
            # use warp to convert the line points to the map coordinates:

            if line_points[0] is None:
                continue

            scale = CALIB_IMAGE_SIZE[0]/480.0
            line_points = np.array(line_points)*scale

            # convert the line points to world coordinates:
            line_points_world = np.array([pixel_to_world((x, y), TRANSFORM) for x, y in line_points])

            # Set current location
            current_location = np.array([0.0, 0.0])

            # Calculate line midpoint
            line_midpoint = (line_points_world[0] + line_points_world[1]) / 2
            line_endpoint = line_points_world[1]

            # Calculate line direction
            line_direction = line_points_world[1] - line_points_world[0]
            line_direction_normalized = line_direction / np.linalg.norm(line_direction)

            # PART 1: Fit spline from current location to line endpoint
            # Control points for the spline
            spline_control_points = np.vstack([
                current_location,
                line_midpoint,
                line_endpoint
            ])

            # Create parameter values for spline (using cumulative distance)
            t = np.zeros(len(spline_control_points))
            for i in range(1, len(spline_control_points)):
                t[i] = t[i-1] + np.linalg.norm(spline_control_points[i] - spline_control_points[i-1])
            t = t / t[-1]  # Normalize to [0,1]

            # Fit the spline
            spline_x = CubicSpline(t, spline_control_points[:, 0])
            spline_y = CubicSpline(t, spline_control_points[:, 1])

            # Generate points along the spline
            num_spline_points = 30
            t_values = np.linspace(0, 1, num_spline_points)
            spline_points = np.column_stack((spline_x(t_values), spline_y(t_values)))

            # PART 2: Create straight extension from line endpoint
            extension_distance = 2.0  # Distance to extend beyond line endpoint
            num_extension_points = 30

            # Generate points along the straight extension
            extension_t = np.linspace(0, 1, num_extension_points)
            extension_points = np.array([
                line_endpoint + t * extension_distance * line_direction_normalized
                for t in extension_t
            ])

            delta_path = np.vstack((spline_points, extension_points))

            # apply the mobots rotation and position to the delta_path:
            # rotate the path by the mobots theta
            theta = image_pose[2]*np.pi/180
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            # rotate the path
            rotated_path = delta_path @ rotation_matrix.T

            # translate the path to the mobots position
            new_path = rotated_path + np.array([image_pose[0], image_pose[1]])

            
            # update the mobot pose
            mobot.set_path(new_path)

            # create the server image:
            # sim_image = locator.render_sim_image(pose=image_pose+delta_pose, cam_image=cam_mask)
            sim_image = np.zeros([270, 480, 3])
            sim_image[:, :, :] = line_image 
                        
            mobot_path_pix = locator.pose_to_pixel(mobot_path)
            map_render = map_renderer.plot_path(mobot_path_pix, image_pose[2]*np.pi/180)
            
            server_image = np.zeros([270*2, 480*2, 3])
            server_image[:270, :480] = cv2.resize(frame, (480, 270))
            server_image[:270, 480:] = sim_image   
            server_image[270:, :] = cv2.resize(map_render, (480*2, 270))             
            
            # Update the frame in the web server
            server.update_frame(server_image)
            
            # Save the frame with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_mmm
            img_path = os.path.join(save_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(img_path, frame)

            # save the server image
            img_path = os.path.join(save_dir, f"server_image_{timestamp}.jpg")
            cv2.imwrite(img_path, server_image)
            
            # Update statistics
            frame_count += 1
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                fps = frame_count / (current_time - last_stats_time)
                print(f"Camera Loop at {fps:.2f} FPS, saved {frame_count} images")
                frame_count = 0
                last_stats_time = current_time
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        mobot.stop()
        cap.release()
        server.stop()
        print("Resources released and server stopped")
        

    input("press enter to stop")
    plt.figure()
    plt.plot(mobot.xs, mobot.ys)
    plt.plot(mobot.path[:, 0], mobot.path[:, 1])
    # save the plot
    plt.savefig('test_plot2.png')
    print('killed')



if __name__ == "__main__":
    # encoder_test(17)
    # input('hit enter to stop')
    # test3()
    test_simple_path()


