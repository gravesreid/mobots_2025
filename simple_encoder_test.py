import lgpio
import time

chip = 4  # GPIO chip based on gpioinfo

# Encoder 1 GPIO pins
gpioA1 = 20
gpioB1 = 16

# Encoder 2 GPIO pins
gpioA2 = 18
gpioB2 = 17

# Open the GPIO chip
h = lgpio.gpiochip_open(chip)

# Set GPIO pins as inputs
lgpio.gpio_claim_input(h, gpioA1)
lgpio.gpio_claim_input(h, gpioB1)
lgpio.gpio_claim_input(h, gpioA2)
lgpio.gpio_claim_input(h, gpioB2)

# Claim alerts on the GPIO pins
lgpio.gpio_claim_alert(h, gpioA1, lgpio.RISING_EDGE)
lgpio.gpio_claim_alert(h, gpioB1, lgpio.RISING_EDGE)
lgpio.gpio_claim_alert(h, gpioA2, lgpio.RISING_EDGE)
lgpio.gpio_claim_alert(h, gpioB2, lgpio.RISING_EDGE)

# Encoder state and positions
encoders = {
    "encoder1": {"last_A": 0, "last_B": 0, "position": 0, "gpioA": gpioA1, "gpioB": gpioB1, "flip": -1},
    "encoder2": {"last_A": 0, "last_B": 0, "position": 0, "gpioA": gpioA2, "gpioB": gpioB2, "flip": 1},
}

def callback(chip, gpio, level, timestamp):
    global encoders

    for encoder_name, encoder in encoders.items():
        if gpio == encoder["gpioA"]:
            encoder["last_A"] = level
            print(f'last_A level: {level}')
        elif gpio == encoder["gpioB"]:
            encoder["last_B"] = level
            print(f'last_B level: {level}')

        # Determine direction based on state transitions
        if encoder["last_A"] == 1 and encoder["last_B"] == 0:
            encoder["position"] += encoder["flip"]  # Adjust for flipped encoder
        elif encoder["last_A"] == 0 and encoder["last_B"] == 1:
            encoder["position"] -= encoder["flip"]  # Adjust for flipped encoder


    print(f"Encoder 1 Position: {encoders['encoder1']['position']}, "
          f"Encoder 2 Position: {encoders['encoder2']['position']}")

# Register callbacks for GPIO pins
lgpio.callback(h, gpioA1, lgpio.BOTH_EDGES, callback)
lgpio.callback(h, gpioB1, lgpio.BOTH_EDGES, callback)
lgpio.callback(h, gpioA2, lgpio.BOTH_EDGES, callback)
lgpio.callback(h, gpioB2, lgpio.BOTH_EDGES, callback)

try:
    print("Monitoring encoder signals...")
    time.sleep(10)  # Monitor for 10 seconds
finally:
    lgpio.gpiochip_close(h)
    print("Cleaned up GPIO resources.")


