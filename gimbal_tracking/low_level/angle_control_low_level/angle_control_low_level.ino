/**
 *
 * Position/angle motion control example
 * Steps:
 * 1) Configure the motor and magnetic sensor
 * 2) Run the code
 * 3) Set the target angle (in radians) from serial terminal
 *
 */
#include <SimpleFOC.h>
#include <math.h>

// magnetic sensor instance - SPI
MagneticSensorSPI sensor = MagneticSensorSPI(AS5147_SPI, 10);
// magnetic sensor instance - MagneticSensorI2C
//MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);
// magnetic sensor instance - analog output
// MagneticSensorAnalog sensor = MagneticSensorAnalog(A1, 14, 1020);

// BLDC motor & driver instance
BLDCMotor motor = BLDCMotor(14);
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 5, 6, 8);
// Stepper motor & driver instance
//StepperMotor motor = StepperMotor(50);
//StepperDriver4PWM driver = StepperDriver4PWM(9, 5, 10, 6,  8);

// angle set point variable
float target_angle = 3.86;
float offset_angle = 3.86;
float motor_position = 0.0;
// instantiate the commander
Commander command = Commander(Serial);
void doTarget(char* cmd) {
  command.scalar(&target_angle, cmd);
}

uint8_t init_position = 0;
unsigned long last_send_time = 0;
const unsigned long send_interval = 1000;  // 50ms

void setup() {

  // use monitoring with serial
  Serial.begin(115200);
  // enable more verbose output for debugging
  // comment out if not needed
  SimpleFOCDebug::enable(&Serial);

  // initialise magnetic sensor hardware
  sensor.init();
  // link the motor to the sensor
  motor.linkSensor(&sensor);

  // driver config
  // power supply voltage [V]
  driver.voltage_power_supply = 24;
  driver.init();
  // link the motor and the driver
  motor.linkDriver(&driver);

  // choose FOC modulation (optional)
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;

  // set motion control loop to be used
  motor.controller = MotionControlType::angle;

  // contoller configuration
  // default parameters in defaults.h

  // velocity PI controller parameters
  motor.PID_velocity.P = 0.2f;
  motor.PID_velocity.I = 20;
  motor.PID_velocity.D = 0;
  // maximal voltage to be set to the motor
  motor.voltage_limit = 12;

  // velocity low pass filtering time constant
  // the lower the less filtered
  motor.LPF_velocity.Tf = 0.01f;

  // angle P controller
  motor.P_angle.P = 20;
  motor.P_angle.D = 0.5;
  // motor.P_angle.output_ramp = 10000; // default 1e6 rad/s^2
  // maximal velocity of the position control
  motor.velocity_limit = 20;
  // comment out if not needed
  // motor.useMonitoring(Serial);
  // initialize motor
  motor.init();
  // align sensor and start FOC
  motor.initFOC();
  // add target command T
  command.add('T', doTarget, "target angle");
}


void loop() {
  motor.loopFOC();
  float current_angle = sensor.getAngle();
  motor_position = current_angle - offset_angle;


  if (millis() - last_send_time >= send_interval) {
    Serial.print("p");
    Serial.println(motor_position);
    last_send_time = millis();
  }


  if (init_position == 0) {
    motor.move(target_angle);
    if (abs(motor_position) < 0.01) {
      init_position = 1;
      target_angle = 0.0;
    }
  } else {
    // Normalize giữ đúng dấu âm/dương
    if (target_angle >= 2 * PI || target_angle <= -2 * PI) {
      target_angle = fmod(target_angle, 2 * PI);
    }

    if (motor_position >= 2 * PI || motor_position <= -2 * PI) {
      target_angle = 0.0;
    }

    motor.move(target_angle + offset_angle);
    command.run();
  }
} 