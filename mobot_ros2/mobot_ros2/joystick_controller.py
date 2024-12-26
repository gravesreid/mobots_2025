import rclpy
from rclpy.node import Node
import RPi.GPIO as GPIO
from sensor_msgs.msg import Joy

class MotorController(Node):

    def __init__(self):
        super().__init__('motor_controller')

        # GPIO Setup
        GPIO.setmode(GPIO.BOARD)

        # Motor 1
        self.ENA1, self.IN1, self.IN2 = 33, 31, 29
        GPIO.setup(self.ENA1, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        self.PWMA = GPIO.PWM(self.ENA1, 100)
        self.PWMA.start(0)

        # Motor 2
        self.ENA2, self.IN3, self.IN4 = 32, 18, 16
        GPIO.setup(self.ENA2, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        self.PWMB = GPIO.PWM(self.ENA2, 100)
        self.PWMB.start(0)

        # Joy Subscriber
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

    def joy_callback(self, msg):
        # Map joystick axes to motor control
        # Assuming axes[0] controls left-right, axes[1] controls forward-backward
        left_motor_speed = int((msg.axes[1] + msg.axes[0]) * 50)  # Adjust scaling as necessary
        right_motor_speed = int((msg.axes[1] - msg.axes[0]) * 50)

        self.control_motor(self.PWMA, self.IN1, self.IN2, left_motor_speed)
        self.control_motor(self.PWMB, self.IN3, self.IN4, right_motor_speed)

    def control_motor(self, pwm, in1, in2, speed):
        if speed > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(abs(speed))

    def destroy_node(self):
        # Cleanup GPIO on shutdown
        self.PWMA.stop()
        self.PWMB.stop()
        GPIO.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    try:
        rclpy.spin(motor_controller)
    except KeyboardInterrupt:
        pass
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
