import time
import lgpio

# Define GPIO pins for Motor 1 and Motor 2
ENA1, IN1, IN2 = 13, 6, 5  # Replace with actual GPIO offsets for Motor 1
ENA2, IN3, IN4 = 12, 24, 23 # Replace with actual GPIO offsets for Motor 2

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

# Open the GPIO chip and claim pins
h = lgpio.gpiochip_open(4)  # Adjust the chip number as needed
lgpio.gpio_claim_output(h, ENA1)
lgpio.gpio_claim_output(h, IN1)
lgpio.gpio_claim_output(h, IN2)
lgpio.gpio_claim_output(h, ENA2)
lgpio.gpio_claim_output(h, IN3)
lgpio.gpio_claim_output(h, IN4)

try:
    # Drive Motor 1 forward
    print("Driving Motor 1 forward...")
    drive_motor(h, ENA1, IN1, IN2, duty_cycle=30, duration=5, direction="forward")

    # Drive Motor 1 reverse
    print("Driving Motor 1 reverse...")
    drive_motor(h, ENA1, IN1, IN2, duty_cycle=30, duration=5, direction="reverse")

    # Drive Motor 2 forward
    print("Driving Motor 2 forward...")
    drive_motor(h, ENA2, IN3, IN4, duty_cycle=30, duration=5, direction="forward")

    # Drive Motor 2 reverse
    print("Driving Motor 2 reverse...")
    drive_motor(h, ENA2, IN3, IN4, duty_cycle=30, duration=5, direction="reverse")

except KeyboardInterrupt:
    print("\nStopping motors...")
finally:
    # Ensure motors are off and clean up
    lgpio.gpio_write(h, ENA1, 0)
    lgpio.gpio_write(h, ENA2, 0)
    lgpio.gpiochip_close(h)
    print("GPIO chip closed.")
