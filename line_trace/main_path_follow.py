import lgpio
import time
import numpy as np
from typing import Tuple, Union, Dict, List
import os
import sys
sys.path.append("..")
from headless_visualization import SimpleStreamServer
import cv2
from datetime import datetime
import threading
from camera import main as camera_main, get_line_state


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

    # def set_goal(self, x:float, y:float, theta:float):
    #     self.goal_x = x
    #     self.goal_y = y
    #     self.goal_theta = theta
    #     self.closed_loop_control()
        # self.follow_path_control()

    # def closed_loop_control(self):
    #     # PID coefficients
    #     Kp_theta = 0.1
    #     Ki_theta = 0
    #     Kd_theta = 0.01

    #     Kp_dist = 0.25
    #     Ki_dist = 0
    #     Kd_dist = 0

    #     goal = np.array([self.goal_x, self.goal_y])
    #     curr = np.array([self.x, self.y])
    #     dist_error = np.linalg.norm(goal - curr)

    #     # Initialize persistent state if not already done

    #     # Timing for derivative/integral
    #     current_time = time.time()
    #     dt = current_time - self._last_time if current_time != self._last_time else 1e-6
    #     dt = max(dt, 0.25) # max time step of 0.25s. If its longer, there is probably a problem

    #     # Compute heading error
    #     desired_theta = np.arctan2(goal[1] - curr[1], goal[0] - curr[0])
    #     theta_error = desired_theta - self.theta
    #     theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error)) * min(dist_error, 1.0)

    #     # PID terms for heading
    #     self._integral_theta += theta_error * dt
    #     derivative_theta = (theta_error - self._last_theta_error) / dt

    #     # PID terms for distance
    #     self._integral_dist += dist_error * dt
    #     derivative_dist = (dist_error - self._last_dist_error) / dt

    #     # Compute PWM signals
    #     theta_term = (
    #         Kp_theta * theta_error +
    #         Ki_theta * self._integral_theta +
    #         Kd_theta * derivative_theta
    #     )
    #     dist_term = (
    #         Kp_dist * dist_error +
    #         Ki_dist * self._integral_dist +
    #         Kd_dist * derivative_dist
    #     )

    #     pwm1 = theta_term + dist_term
    #     pwm2 = -theta_term + dist_term

    #     # Update memory
    #     self._last_theta_error = theta_error
    #     self._last_dist_error = dist_error
    #     self._last_time = current_time

    #     self.set_motor_pwms((pwm1, pwm2))

    def set_path(self, path:np.ndarray):
        self.path = path
        self.path_idx = 0
        self.last_time = time.time()
        self.follow_path_control()


    def follow_path_control(self):
        if self.path_idx >= len(self.path) - 2:
            print('reached end of path')
            self.set_motor_pwms((0, 0))
            return
        GOAL_SPEED = 1 # m/s
    
        KP_ANGLE = 0.25
        KI_ANGLE = 0
        KD_ANGLE = 0.01

        KP_SPEED = 0.5
        KI_SPEED = 0

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
            if self.path_idx == len(self.path) - 3:
                self.path_idx += 1 # move to the last point
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
        

import matplotlib.pyplot as plt
from optimze import MobotLocator

from image_thresh import thresh_image
def test_simple_path():
    chip = lgpio.gpiochip_open(4)

    # Start camera in a separate thread
    camera_thread = threading.Thread(target=camera_main)
    camera_thread.daemon = True
    camera_thread.start()

    # Wait for camera to initialize
    time.sleep(2)

    # Initialize robot
    mobot = MoBot(chip=chip, verbose=False)
    
    # PID controller parameters
    k_p = 0.2  # Proportional gain
    k_i = 0.05  # Integral gain
    k_d = 0.1   # Derivative gain
    
    # Base speed - adjust based on your robot's capabilities
    base_speed = 0.2  # Start slower for safety
    
    # PID state variables
    error_integral = 0
    last_error = 0
    last_time = time.time()
    
    # Last valid center line angle
    last_angle = 0
    
    # Initialize motor PWM values
    left_pwm = base_speed
    right_pwm = base_speed
    
    try:
        # Main control loop
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Limit dt to avoid large jumps after pauses
            dt = min(dt, 0.1)
            
            # Check if line is found and get center line angle
            line_found, center_angle = get_line_state()
            
            if line_found:
                # Use the actual angle from the camera
                angle = center_angle
                
                # Calculate steering error
                # The angle represents deviation from center
                # Positive angle means the line is to the right (need to turn right)
                # Negative angle means the line is to the left (need to turn left)
                error = angle
                
                # PID control
                error_integral += error * dt
                error_integral = np.clip(error_integral, -1.0, 1.0)  # Anti-windup
                
                error_derivative = (error - last_error) / dt if dt > 0 else 0
                last_error = error
                
                # Calculate steering command using PID
                steering = k_p * error + k_i * error_integral + k_d * error_derivative
                
                # Apply steering to motor commands
                left_pwm = base_speed - steering
                right_pwm = base_speed + steering
                
                # Remember last valid angle
                last_angle = angle
                
                print(f"Line found! Angle: {angle:.2f}°, Steering: {steering:.2f}")
            else:
                print("No line found. Continuing on previous path")
                # Gradually reduce the differential to go straighter
                # Robot should go straight when no line is found
                left_pwm = 0.9 * left_pwm + 0.1 * base_speed
                right_pwm = 0.9 * right_pwm + 0.1 * base_speed
            
            # Ensure PWM values are within limits
            left_pwm = np.clip(left_pwm, -0.3, 0.3)
            right_pwm = np.clip(right_pwm, -0.3, 0.3)
            
            # Apply the motor values
            mobot.set_motor_pwms((left_pwm, right_pwm))
            
            # Print current motor values for debugging
            print(f"Motors: L={left_pwm:.2f}, R={right_pwm:.2f}")
            
            # Small delay to prevent tight loop
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        mobot.stop()
        print("Resources released")

if __name__ == "__main__":
    # encoder_test(17)
    # input('hit enter to stop')
    # test3()
    test_simple_path()


