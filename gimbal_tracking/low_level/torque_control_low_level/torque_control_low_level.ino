#include <SimpleFOC.h>

MagneticSensorSPI sensor = MagneticSensorSPI(AS5147_SPI, 10);
BLDCMotor motor = BLDCMotor(14);
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 5, 6, 8);
float target_voltage = 0;

float target_angle = 3.9;
float offset_angle = 3.9;
float motor_position = 0.0;
// PID parameters
float Kp = 8.0;
float Ki = 0.0;
float Kd = 0.1;
// State variables
float error_prev = 0;
float integral = 0;
unsigned long last_time = 0;
float max_voltage = 20;

uint8_t init_position = 0;
unsigned long last_send_time = 0;
const unsigned long send_interval = 1000;  // 50ms

const byte BUFFER_SIZE = 20;      // Predefined buffer size
char receivedChars[BUFFER_SIZE];  // Buffer to store received data
boolean newData = false;          // Flag to indicate new data has been received
uint8_t ndx = 0;
unsigned long lastDataReceivedTime = 0;  // Time of last data reception
float targetValue = 0.0;
boolean time_out = false;
const unsigned long timeout_duration = 10000;  // Timeout period in milliseconds (10 seconds)
float timeout_angle = 0.0;                     // Angle to hold during timeout

// PID Position function
float pidPosition(float setpoint, float position_now) {
  unsigned long now = millis();
  float dt = (now - last_time) / 1000.0;  // Convert to seconds
  last_time = now;
  float error = setpoint - position_now;
  integral += error * dt;
  float derivative = (error - error_prev) / dt;
  float output = Kp * error + Ki * integral + Kd * derivative;
  error_prev = error;
  output = constrain(output, -max_voltage, max_voltage);
  return output;
}

void setup() {
  Serial.begin(115200);
  sensor.init();
  motor.linkSensor(&sensor);
  driver.voltage_power_supply = 24;
  driver.voltage_limit = 20.0;
  driver.init();
  motor.linkDriver(&driver);
  motor.voltage_sensor_align = 18;
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::torque;
  motor.init();
  motor.initFOC();
}

void serialEvent() {
  char rc;
  while (Serial.available() > 0) {
    rc = Serial.read();
    if (rc != '\n') {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= BUFFER_SIZE) {
        ndx = BUFFER_SIZE - 1;
      }
    } else {
      receivedChars[ndx] = '\0';
      if (receivedChars[0] == 'T') {
        targetValue = atof(&receivedChars[1]);  // Skip 'T' character
        lastDataReceivedTime = millis();        // Update last data reception time
      }
      ndx = 0;
      time_out = false;
    }
  }
}

void loop() {
  motor.loopFOC();
  float current_angle = sensor.getAngle();
  motor_position = current_angle - offset_angle;

  if (abs(motor_position) > 2 * PI)
    init_position = 0;

  if (init_position == 0) {
    target_voltage = pidPosition(target_angle, current_angle);
    motor.move(target_voltage);
    if (abs(motor_position) < 0.1) {
      init_position = 1;
    }
  } else {
    if (millis() - last_send_time >= send_interval) {
      Serial.print("p");
      Serial.println(motor_position);
      last_send_time = millis();
    }

    // Check for timeout and capture the current angle
    if (millis() - lastDataReceivedTime >= timeout_duration && time_out == false) {
      timeout_angle = current_angle;  // Capture the current angle at timeout
      time_out = true;
    }
    if (time_out) {
      // Use PID to hold the motor at the timeout angle
      target_voltage = pidPosition(timeout_angle, current_angle);
      motor.move(target_voltage);
    } else {
      // Normal operation: use targetValue from serial input
      targetValue = constrain(targetValue, -max_voltage, max_voltage);
      motor.move(targetValue);
    }
  }
}
