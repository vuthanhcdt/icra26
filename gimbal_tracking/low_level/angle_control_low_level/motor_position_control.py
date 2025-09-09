import serial
import time
import threading

# Serial port configuration 
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
TIMEOUT = 1

class MotorController:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=TIMEOUT)
        self.target_angle = 0.0
        self.current_position = 0.0
        self.running = True
        self.lock = threading.Lock()

        print("Arduino connected. Waiting for data...")

    def send_target_angle(self):
        """Send the target angle to Arduino using Commander format."""
        with self.lock:
            command = f"T {self.target_angle:.4f}\n"
            self.ser.write(command.encode('utf-8'))
            print(f"[SEND] {command.strip()}")

    def read_serial(self):
        """Read and process serial data from Arduino."""
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("p"):
                    pos_str = line[1:]
                    self.current_position = float(pos_str)
                    print(f"[READ] Position = {self.current_position:.4f} rad")
            except (ValueError, UnicodeDecodeError):
                continue

    def run(self):
        """Main control loop."""
        serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        serial_thread.start()

        try:
            while self.running:
                self.send_target_angle()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def set_target_angle(self, angle):
        """Set a new target angle."""
        with self.lock:
            self.target_angle = angle
            print(f"[CMD] New target angle: {self.target_angle:.4f} rad")

    def stop(self):
        """Stop the controller and close serial port."""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("Controller stopped.")

def main():
    controller = MotorController(SERIAL_PORT, BAUD_RATE)

    controller.set_target_angle(3)

    control_thread = threading.Thread(target=controller.run)
    control_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop()


if __name__ == "__main__":
    main()
