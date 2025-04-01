import lgpio
import time
from multiprocessing import Process, Queue
import numpy as np
from typing import Tuple, Union, Dict, List

class SingleEncoder:
    def __init__(self, chip, gpio:int, tick_val = 1/17, verbose = False, edge_type = "both"):
        self.chip = chip
        self.gpio = gpio
        self.verbose = verbose

        self.count = 0

        self.tick_val = tick_val
        
        lgpio.gpio_claim_input(self.chip, gpio)
        print('starting encoder with pin', gpio)

        # Claim alerts on the GPIO pins
        if edge_type == "both":
            lgpio.gpio_claim_alert(self.chip, gpio, lgpio.BOTH_EDGES)
            lgpio.callback(chip, gpio, lgpio.BOTH_EDGES, self.callback)
        elif edge_type == "rising":
            lgpio.gpio_claim_alert(self.chip, gpio, lgpio.RISING_EDGE)
            lgpio.callback(chip, gpio, lgpio.RISING_EDGE, self.callback)
        else:
            raise ValueError(f"edge_type {edge_type} not supported")

        self.closed = False

    def callback(self, chip, gpio, level, timestamp):
        self.count += 1
        
        if self.verbose:
            print(f"Encoder {self.gpio}: {self.count}, {self.get_distance()}")

    def get_distance(self):
        return self.count*self.tick_val
    
    def reset(self):
        self.count = 0


class Motor:
    def __init__(self, chip, ena:int, in1:int, in2:int, pwm_period:float):
        self.ena = ena
        self.in1 = in1
        self.in2 = in2
        self.pwm_period = pwm_period
        self.command_queue = Queue()
        self.process = Process(target=self._run_motor, args=(chip, ena, in1, in2, pwm_period, self.command_queue))
        self.process.start()
        self.stopped = False

    def set_speed(self, speed:float):
        if isinstance(speed, int):
            speed = float(speed)
        assert isinstance(speed, float) and speed >= -1 and speed <= 1, "speed must be a float between -1.0 and 1.0"
        self.command_queue.put(speed)

    def stop(self):
        self.command_queue.put("stop")
        self.process.join()
        self.stopped = True

    def __del__(self):
        if not self.stopped:
            self.stop()

    @staticmethod
    def _run_motor(chip,
                ena:int,
                in1:int,
                in2:int,
                pwm_period:float,
                command_queue: Queue
                ):

        lgpio.gpio_claim_output(chip, ena)
        lgpio.gpio_claim_output(chip, in1)
        lgpio.gpio_claim_output(chip, in2)

        last_low_time = time.time()

        command = 0.0
        current_dir = 0 # set the initial direction to 1 (forward)

        while True:
            
            # get the last command that was send, discard the rest. 
            # if no command was set, it will continue running based on
            # the last command (command is just not updated).
            while not command_queue.empty():
                command = command_queue.get()

            if command == "stop":
                break
            
            high_time = abs(command) * pwm_period

            # Set direction
            if command >= 0: # forward
                if current_dir != 1: # if we are not going the right direction, change direction
                    lgpio.gpio_write(chip, in1, 1)
                    lgpio.gpio_write(chip, in2, 0)
                    current_dir = 1
            else: # backward
                if current_dir != -1:
                    lgpio.gpio_write(chip, in1, 0)
                    lgpio.gpio_write(chip, in2, 1)
                    current_dir = -1
                    
            lgpio.gpio_write(chip, ena, 1)  # Enable high
            time.sleep(high_time)
            lgpio.gpio_write(chip, ena, 0)  # Enable low
            low_time = pwm_period - (time.time() - last_low_time)
            if low_time > 0:
                time.sleep(low_time)
            last_low_time = time.time()
        

def drive_motor(h, ena, in1, in2, duty_cycle, duration, direction):
    pwm_period = 0.01  # Simulated PWM period (10ms)
    high_time = duty_cycle / 100 * pwm_period
    low_time = pwm_period - high_time
    end_time = time.time() + duration

    # Set direction
    if direction == "forward":
        lgpio.gpio_write(h, in1, 1)
        lgpio.gpio_write(h, in2, 0)
    elif direction == "reverse":
        lgpio.gpio_write(h, in1, 0)
        lgpio.gpio_write(h, in2, 1)

    # Simulate PWM
    while time.time() < end_time:
        lgpio.gpio_write(h, ena, 1)  # Enable high
        time.sleep(high_time)
        lgpio.gpio_write(h, ena, 0)  # Enable low
        time.sleep(low_time)

def test1():
    ENA1, IN1, IN2 = 13, 6, 5  # Motor 1
    ENA2, IN3, IN4 = 12, 24, 23  # Motor 2

    chip = lgpio.gpiochip_open(4)
    encoder1 = SingleEncoder(chip = chip,
                               gpio = 17,
                               verbose=True)

    encoder2 = SingleEncoder(chip = chip,
                               gpio=20,
                               verbose=True)
    
    lgpio.gpio_claim_output(chip, ENA1)
    lgpio.gpio_claim_output(chip, IN1)
    lgpio.gpio_claim_output(chip, IN2)
    lgpio.gpio_claim_output(chip, ENA2)
    lgpio.gpio_claim_output(chip, IN3)
    lgpio.gpio_claim_output(chip, IN4)

    # Run motor tests while reading encoder values
    # print("Driving Motor 1 forward...")
    # drive_motor(chip, ENA1, IN1, IN2, duty_cycle=10, duration=5, direction="forward")

    # print("Driving Motor 1 reverse...")
    # drive_motor(chip, ENA1, IN1, IN2, duty_cycle=10, duration=5, direction="reverse")
    # print("Monitoring encoder signals...")

    print("Driving Motor 2 forward...")
    drive_motor(chip, ENA2, IN3, IN4, duty_cycle=10, duration=5, direction="forward")

    print("Driving Motor 2 reverse...")
    drive_motor(chip, ENA2, IN3, IN4, duty_cycle=10, duration=5, direction="reverse")

def test2():
    ENA1, IN1, IN2 = 13, 6, 5  # Motor 1
    encoder_pin1 = 20
    ENA2, IN3, IN4 = 12, 24, 23  # Motor 2
    encoder_pin2 = 17

    chip = lgpio.gpiochip_open(4)
    motor1 = Motor(chip = chip,
                   ena=ENA1,
                   in1=IN1,
                   in2=IN2,
                   pwm_period=0.01)

    encoder1 = SingleEncoder(chip=chip,
                             gpio=encoder_pin1,
                             verbose=False)
    
    motor2 = Motor(chip=chip,
                   ena = ENA2,
                   in1 = IN3,
                   in2 = IN4,
                   pwm_period = 0.01)
    
    encoder2 = SingleEncoder(chip=chip,
                             gpio=encoder_pin2,
                             verbose=False)

    print('motor 1 forward')
    motor1.set_speed(0.1)
    time.sleep(5)
    print('motor 2 forward')
    motor1.set_speed(0)
    motor2.set_speed(0.1)
    time.sleep(5)
    motor1.stop()
    motor2.stop()
    print('done')


def test3():
    ENA1, IN1, IN2 = 13, 6, 5  # Motor 1
    encoder_pin1 = 20
    ENA2, IN3, IN4 = 12, 24, 23  # Motor 2
    encoder_pin2 = 17

    chip = lgpio.gpiochip_open(4)
    motor1 = Motor(chip = chip,
                   ena=ENA1,
                   in1=IN1,
                   in2=IN2,
                   pwm_period=0.01)

    encoder1 = SingleEncoder(chip=chip,
                             gpio=encoder_pin1,
                             verbose=False)
    
    motor2 = Motor(chip=chip,
                   ena = ENA2,
                   in1 = IN3,
                   in2 = IN4,
                   pwm_period = 0.01)
    
    encoder2 = SingleEncoder(chip=chip,
                             gpio=encoder_pin2,
                             verbose=False)

    print('forward')
    motor1.set_speed(0.2)
    motor2.set_speed(-0.2)
    time.sleep(5)
    print('stop')
    motor1.set_speed(0)
    motor2.set_speed(0)
    time.sleep(5)
    print('kill')
    motor1.stop()
    motor2.stop()
    print('done')
    print("encoder1", encoder1.get_distance())
    print('encoder2', encoder2.get_distance())

def encoder_calib():
    encoder_pin1 = 20
    encoder_pin2 = 17
    chip = lgpio.gpiochip_open(4)
    encoder1 =  SingleEncoder(chip=chip,
                             gpio=encoder_pin1,
                             verbose=True,
                             tick_val=0.009)
    
    encoder2 =  SingleEncoder(chip=chip,
                             gpio=encoder_pin2,
                             verbose=True,
                             tick_val=0.009)
    
    print('begin')
    a = input('press enter when done')
    print('a', a)
    print('encoder 1:', encoder1.get_distance())
    print('encoder 2:', encoder2.get_distance())

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
        W_LOOK = 0.2 # rad/s

        KP_ANGLE = 0.05
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

        goal_heading = W_LOOK * look_heading + (1 - W_LOOK) * path_heading

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
def test_mobot():
    chip = lgpio.gpiochip_open(4)
    mobot = MoBot(chip=chip, verbose=True)
    mobot.set_goal(x=1, y=1, theta=0)
    input("press enter to stop")
    mobot.stop()
    # plot the path and the robot's path
# def test_simple_path():
#     chip = lgpio.gpiochip_open(4)
#     mobot = MoBot(chip=chip, verbose=True)
#     # load path from csv "simple_path.csv", in the form x,y,t
#     path = np.loadtxt("simple_path.csv", delimiter=",", skiprows=1)
#     start_time = time.time()
#     for x, y, t in path:
#         delay = t - (time.time() - start_time)
#         if delay > 0:
#             time.sleep(delay)
#         mobot.set_goal(x, y, 0)
#     mobot.stop()
#     print('finished path!')

def test_simple_path():
    chip = lgpio.gpiochip_open(4)
    mobot = MoBot(chip=chip, verbose=True)
    # load path from csv "simple_path.csv", in the form x,y,t
    path = np.loadtxt("simple_path.csv", delimiter=",", skiprows=1)
    mobot.set_path(path)
    input("press enter to stop")
    plt.figure()
    plt.plot(mobot.xs, mobot.ys)
    plt.plot(mobot.path[:, 0], mobot.path[:, 1])
    # save the plot
    plt.savefig('test_plot.png')
    print('killed')



if __name__ == "__main__":
    # encoder_test(17)
    # input('hit enter to stop')
    # test3()
    test_simple_path()


