# Imported Libraries
import RPi.GPIO as GPIO
from time import sleep

# SetMode
GPIO.setmode(GPIO.BOARD)

# Motor 1 Setup
PWR1, ENA1, IN1, IN2, GND = 2, 33, 31, 29, 39
GPIO.setup(ENA1, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
PWMA = GPIO.PWM(ENA1, 100)
PWMA.start(0)

# Motor 2 Setup
PWR2, ENA2, IN3, IN4, GND = 4, 32, 18, 16, 34
GPIO.setup(ENA2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
PWMB = GPIO.PWM(ENA2, 100)
PWMB.start(0)

# Motor 1 Drive
PWMA.ChangeDutyCycle(30)
sleep(5)
GPIO.output(IN1, GPIO.HIGH)
sleep(5)
GPIO.output(IN2, GPIO.LOW)

# Motor 2 Drive
PWMB.ChangeDutyCycle(30)
sleep(5)
GPIO.output(IN3, GPIO.LOW)
sleep(5)
GPIO.output(IN4, GPIO.HIGH)

# Cleanup
GPIO.cleanup()
