import lgpio
import time
import numpy as np
from typing import Tuple, Union, Dict, List
import os
from headless_visualization import SimpleStreamServer
import cv2
from datetime import datetime

class MapRenderer():
    def __init__(self, map_path:str, line_step:int = 10):
        self.base_map = cv2.imread(map_path)
        self.path_idx = 0
        self.line_step = line_step

    def plot_path(self, path:np.ndarray):
        # update the line from the last point to the current point
        for i in range(self.path_idx, len(path) - 1, self.line_step):
            cv2.line(self.base_map, tuple(path[i]), tuple(path[i+1]), (0, 0, 255), 2)

        self.path_idx = i + 1
        
        map_image = self.base_map.copy()
        # add the last point
        cv2.circle(map_image, tuple(path[-1]), 5, (255, 0, 0), -1)

        return map_image



class MoBot():
    ENA1, IN1_A, IN1_B = 13, 6, 5  # Motor 1
    ENCODER1 = 20
    ENA2, IN2_A, IN2_B = 12, 24, 23  # Motor 2
    ENCODER2 = 17
    TICK_DIST = 0.009
    WIDTH = 0.23
    PWM_PERIOD = 0.01

    MOTOR1_GAIN = 1
    MOTOR2_GAIN = -1

    MOTOR1_LIM = [0, 0.2]
    MOTOR2_LIM = [-0.2, 0]

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
        self._last_pos = np.array([0, 0])

        self._last_time = time.time()

        self.path_idx = 0
        self.path = np.array([[0, 0]])

        self.xs = []
        self.ys = []
        self.thetas = []

    
    def callback(self, chip, gpio, level, timestamp):
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
        pwm1 = np.clip(pwms[0] * self.MOTOR1_GAIN, self.MOTOR1_LIM[0], self.MOTOR1_LIM[1])
        pwm2 = np.clip(pwms[1] * self.MOTOR2_GAIN, self.MOTOR2_LIM[0], self.MOTOR2_LIM[1])

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
        lgpio.tx_pwm(self.chip, self.ENA1, 0, 0)
        lgpio.tx_pwm(self.chip, self.ENA2, 0, 0)

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
        LOOK_AHEAD = 1 # m
        DEVIATION_THRESH = 1 # m - the relative importance of moving back to the line. W_LOOK = min(distance_from_line / DEVIATION_THRESH, 1)

        KP_ANGLE = 0.25
        KI_ANGLE = 0
        KD_ANGLE = 0.05

        KP_SPEED = 0.25
        KI_SPEED = 0

        dt = time.time() - self.last_time
        dt = max(dt, 0.25) # max time step of 0.25s. If its longer, there is probably a problem
        pos = np.array([self.x, self.y])

        speed = np.linalg.norm(self._last_pos - pos) / dt
        speed = np.clip(speed, 0, 10) # clip the speed to 10 m/s. It should not be more than that

        speed_error = GOAL_SPEED - speed
        self._integral_speed += speed_error * dt


        # only move forward in the path. Check to see if we are closer to the next point
        # if we are, move to the next point
        while True:
            if self.path_idx == len(self.path) - 3:
                self.path_idx += 1 # move to the last point
                break
            cur_dist = np.linalg.norm(self.path[self.path_idx] - pos)
            next_dist = np.linalg.norm(self.path[self.path_idx + 1] - pos)

            if next_dist < cur_dist:
                self.path_idx += 1
            else:  
                break

        print('path idx:', self.path_idx)

        # get the path heading:
        tangent = self.path[self.path_idx + 1] - self.path[self.path_idx - 1]
        path_heading = np.arctan2(tangent[1], tangent[0])

        # calculate the distance from the path (perpendicular distance)
        # the distance is the cross product of the vector from the current point to the robot
        # and the tangent vector of the path (unit vector)

        dist = np.cross(pos - self.path[self.path_idx], tangent / np.linalg.norm(tangent))
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
        tangent = self.path[look_idx] - self.path[look_idx - 1]
        look_heading = np.arctan2(tangent[1], tangent[0])

        w_look = min(np.abs(dist) / DEVIATION_THRESH, 1)
        goal_heading = w_look * look_heading + (1 - w_look) * path_heading

        angle_error = goal_heading - self.theta

        # wrap the angle error
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        angle_error_dot = (angle_error - self._last_theta_error) / dt
        self._integral_theta += angle_error * dt

        pwm_angle = KP_ANGLE * angle_error + KI_ANGLE * self._integral_theta + KD_ANGLE * angle_error_dot
        pwm_speed = KP_SPEED * speed_error + KI_SPEED * self._integral_speed

        pwm_1 = pwm_speed + pwm_angle
        pwm_2 = pwm_speed - pwm_angle

        self.set_motor_pwms((pwm_1, pwm_2))
        

import matplotlib.pyplot as plt
from optimze import MobotLocator
from image_thesh import thresh_image
def test_simple_path():
    chip = lgpio.gpiochip_open(4)
    mobot = MoBot(chip=chip, verbose=True)
    # load path from csv "simple_path.csv", in the form x,y,t
    path = np.loadtxt("/home/pi/mobots_2025/map_processing/test_points_processed.csv", delimiter=",", skiprows=1)
    mobot.set_path(path)

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

    locator = MobotLocator(max_detlas=np.array([0.1, 0.1, 10]), step_size=np.array([0.01, 0.01, 1]))
    map_renderer = MapRenderer("/home/pi/mobots_2025/map_processing/final_path.png")
    while mobot.path_idx < len(mobot.path) - 2:
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
                    print("Error: Could not read frame")
                    break

                image_pose = np.array([mobot.x, mobot.y, mobot.theta])

                resized_image = cv2.resize(frame, (480, 270))
                cam_mask = thresh_image(resized_image)  #threshold the image

                #threshold the image


                # run the locator
                delta_pose = locator.locate_image(cam_image=cam_mask, 
                                                  x=image_pose[0], 
                                                  y=image_pose[1], 
                                                  theta=image_pose[2])
                
                # update the mobot pose
                mobot.x += delta_pose[0]
                mobot.y += delta_pose[1]
                mobot.theta += delta_pose[2]

                # create the server image:
                sim_image = locator.render_sim_image(pose=image_pose+delta_pose, cam_image=cam_mask)
                
                mobot_path = np.array([mobot.x, mobot.y]).copy().T
                mobot_path_pix = locator.pose_to_pixel(mobot_path)
                map_render = map_renderer.plot_path(mobot_path_pix)
                
                server_image = np.zeros(480*2, 270*2, 3)
                server_image[:480, :270] = resized_image
                server_image[:480, 270:] = sim_image   
                server_image[480:, :] = cv2.resize(map_render, (480*2, 270))             
                
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


