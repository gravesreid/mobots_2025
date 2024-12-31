import time
import lgpio
from rotary_encoder import decoder

# Define GPIO pins for motors
ENA1, IN1, IN2 = 13, 6, 5  # Motor 1
ENA2, IN3, IN4 = 12, 24, 23  # Motor 2

# Define GPIO pins for encoders
ENC_A1, ENC_B1 = 17, 27  # Encoder for Motor 1
ENC_A2, ENC_B2 = 25, 16  # Encoder for Motor 2

# Position counters for encoders
pos1 = 0
pos2 = 0

# Callbacks for encoders
def callback_motor1(way):
    global pos1
    pos1 += way
    print(f"Motor 1 Encoder Position: {pos1}")

def callback_motor2(way):
    global pos2
    pos2 += way
    print(f"Motor 2 Encoder Position: {pos2}")

# Function to control a motor
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

# Main script
try:
    # Open the GPIO chip and claim pins for motors
    h = lgpio.gpiochip_open(4)
    lgpio.gpio_claim_output(h, ENA1)
    lgpio.gpio_claim_output(h, IN1)
    lgpio.gpio_claim_output(h, IN2)
    lgpio.gpio_claim_output(h, ENA2)
    lgpio.gpio_claim_output(h, IN3)
    lgpio.gpio_claim_output(h, IN4)

    # Initialize encoders
    encoder1 = decoder(lgpio, 4, ENC_A1, ENC_B1, callback_motor1)
    encoder2 = decoder(lgpio, 4, ENC_A2, ENC_B2, callback_motor2)
    print("Encoders initialized.")

    # Run motor tests while reading encoder values
    print("Driving Motor 1 forward...")
    drive_motor(h, ENA1, IN1, IN2, duty_cycle=50, duration=5, direction="forward")

    print("Driving Motor 1 reverse...")
    drive_motor(h, ENA1, IN1, IN2, duty_cycle=50, duration=5, direction="reverse")

    print("Driving Motor 2 forward...")
    drive_motor(h, ENA2, IN3, IN4, duty_cycle=50, duration=5, direction="forward")

    print("Driving Motor 2 reverse...")
    drive_motor(h, ENA2, IN3, IN4, duty_cycle=50, duration=5, direction="reverse")

except KeyboardInterrupt:
    print("\nStopping motors...")

finally:
    # Ensure motors are off and clean up
    lgpio.gpio_write(h, ENA1, 0)
    lgpio.gpio_write(h, ENA2, 0)
    encoder1.cancel()
    encoder2.cancel()
    lgpio.gpiochip_close(h)
    print("GPIO chip closed.")

