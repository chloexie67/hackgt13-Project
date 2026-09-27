/*
  Ball X Position -> X Stepper Angle
  ----------------------------------
  Reads ball data from demo.py over USB Serial (115200 baud), one line at a time:
      X,Y,VX,VY\n
  e.g.  "12.3,-4.8,1.5,-0.2\n"

  Only X (metres from the centre spot along the pitch) is used; velocity is ignored.
  Every 2 m of X is 0.9 degrees of X motor rotation, measured from where the
  motor was at power-on (step 0):
    X = +2 m   -> 0.9 deg one way       X = -2 m   -> 0.9 deg the other way
    X = +10 m  -> 4.5 deg               X = -10 m  -> 4.5 deg the other way
    X = +52.5 m (goal line) -> 23.4 deg X = -52.5 m -> 23.4 deg the other way
  Positive X turns the motor clockwise, negative X counterclockwise (see
  X_POSITIVE_STEPS_ARE_CW if that comes out reversed).

  0.9 deg is one half-step of a 200 steps/rev motor, so the motor is half-stepped
  and moves one half-step per 2 m.

  X motor: 200 steps/rev, 4-wire, on pins 32, 33, 25, 26 (driver IN1-IN4),
  driven with AccelStepper (install "AccelStepper" by Mike McCauley).
  Level the board before powering up: step 0 is wherever it is at power-on.
  Close the Arduino Serial Monitor before running demo.py: only one program
  can use the USB port at a time.
*/

#include <AccelStepper.h>

// ================= X motor =================
const int X_PIN_IN1 = 32;
const int X_PIN_IN2 = 33;
const int X_PIN_IN3 = 25;
const int X_PIN_IN4 = 26;

const float METRES_PER_STEP = 2.0f;     // 2 m of ball X ...
const float DEG_PER_STEP    = 0.9f;     // ... = 0.9 deg = one half-step of a 200 steps/rev motor

const float X_FIELD_MAX = 105.0f / 2.0f;   // +/- 52.5 m; X beyond the goal lines is clamped

const float X_MAX_SPEED    = 200.0f;   // half-steps/s
const float X_ACCELERATION = 100.0f;   // half-steps/s^2

// Which way positive steps turn the motor depends on the wiring:
// set this to false if positive X turns the motor counterclockwise.
const bool X_POSITIVE_STEPS_ARE_CW = true;
const int  X_CW_SIGN = X_POSITIVE_STEPS_ARE_CW ? 1 : -1;

AccelStepper xStepper(AccelStepper::HALF4WIRE, X_PIN_IN1, X_PIN_IN2, X_PIN_IN3, X_PIN_IN4);

// ================= Serial input =================
const unsigned int MAX_LINE_LENGTH = 64;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);

  xStepper.setMaxSpeed(X_MAX_SPEED);
  xStepper.setAcceleration(X_ACCELERATION);
  xStepper.setCurrentPosition(0);

  Serial.println("Ready. Send X,Y,VX,VY: every 2 m of X turns the X motor 0.9 deg");
}

void loop() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      processLine(inputBuffer);
      inputBuffer = "";
    } else if (incomingChar != '\r') {
      if (inputBuffer.length() < MAX_LINE_LENGTH) {
        inputBuffer += incomingChar;
      } else {
        inputBuffer = "";  // garbage without a newline: drop it
      }
    }
  }

  xStepper.run();
}

// ---------------------------------------------------------------
// Reads X from one "X,Y,VX,VY" line and sends the X motor to
// 0.9 deg per 2 m, clockwise for positive X, counterclockwise
// for negative X.
// ---------------------------------------------------------------
void processLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  float values[4];
  if (!parseFourFloats(line, values)) {
    Serial.println("Invalid input. Expected format: X,Y,VX,VY");
    return;
  }

  float ballX = constrain(values[0], -X_FIELD_MAX, X_FIELD_MAX);
  long target = X_CW_SIGN * (long) round(ballX / METRES_PER_STEP);
  xStepper.moveTo(target);

  Serial.print("X ");
  Serial.print(ballX);
  Serial.print(" m -> ");
  Serial.print(target * DEG_PER_STEP * X_CW_SIGN);
  Serial.print(" deg (");
  Serial.print(target);
  Serial.println(" half-steps)");
}

// ---------------------------------------------------------------
// Splits "a,b,c,d" into 4 floats. Returns false on malformed input.
// ---------------------------------------------------------------
bool parseFourFloats(String line, float outValues[4]) {
  int startIndex = 0;
  for (int i = 0; i < 4; i++) {
    int commaIndex = line.indexOf(',', startIndex);
    String token;

    if (i < 3) {
      if (commaIndex == -1) return false;
      token = line.substring(startIndex, commaIndex);
      startIndex = commaIndex + 1;
    } else {
      token = line.substring(startIndex);
    }

    token.trim();
    if (token.length() == 0) return false;
    outValues[i] = token.toFloat();
  }
  return true;
}
