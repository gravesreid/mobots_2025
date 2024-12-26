import serial
import time

# Configure serial communication
SERIAL_PORT = '/dev/ttyUSB0'  # Update this to your port (e.g., COM3 on Windows)
BAUD_RATE = 9600

def main():
    try:
        # Initialize serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(5)  # Allow time for Arduino to reset

        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Type your commands (e.g., F200, B150, S0). Type 'exit' to quit.")

        while True:
            # Get user input
            command = input("Enter command: ").strip()
            if command.lower() == 'exit':
                print("Exiting program.")
                break

            # Send command to Arduino
            ser.write((command + '\n').encode())
            print(f"Sent: {command}")

        ser.close()
        print("Serial connection closed.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted. Closing serial connection.")
        if ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
