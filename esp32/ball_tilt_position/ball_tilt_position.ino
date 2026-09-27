/*
  Ball Velocity -> X Stepper Direction
  ------------------------------------
  Reads ball data from demo.py over USB Serial (115200 baud), one line at a time:
      X,Y,VX,VY\n
  e.g.  "12.3,-4.8,1.5,-0.2\n"

  Only VX (the ball's velocity along the pitch, m/s) is used:
    VX > 0  -> X motor turns to 15 degrees clockwise of its starting position
    VX < 0  -> X motor turns to 15 degrees counterclockwise of its starting position
    VX = 0  -> X motor stays where it is

  The 15 degrees is measured from where the motor was at power-on (step 0), so
  the motor swings between the two sides and never keeps turning further.
  With 1.8 deg steps, 15 degrees rounds to 8 steps (14.4 degrees).

  X motor: 200 steps/rev, 4-wire, on pins 32, 33, 25, 26 (driver IN1-IN4),
  driven with AccelStepper (install "AccelStepper" by Mike McCauley).
  Close the Arduino Serial Monitor before running demo.py: only one program
  can use the USB port at a time.
*/

#include <AccelStepper.h>

// ================= X motor =================
const int X_PIN_IN1 = 32;
const int X_PIN_IN2 = 33;
const int X_PIN_IN3 = 25;
const int X_PIN_IN4 = 26;

const int   X_STEPS_PER_REV = 200;
const float X_DEG_PER_STEP  = 360.0f / X_STEPS_PER_REV;   // 1.8 deg/step

const float TURN_ANGLE_DEG = 15.0f;
const long  TURN_STEPS     = (long) round(TURN_ANGLE_DEG / X_DEG_PER_STEP);   // 8 steps

const float X_MAX_SPEED    = 200.0f;   // steps/s
const float X_ACCELERATION = 100.0f;   // steps/s^2

// Which way positive steps turn the motor depends on the wiring:
// set this to false if the motor turns counterclockwise for a positive VX.
const bool X_POSITIVE_STEPS_ARE_CW = true;
const long CLOCKWISE_TARGET        = X_POSITIVE_STEPS_ARE_CW ? TURN_STEPS : -TURN_STEPS;
const long COUNTERCLOCKWISE_TARGET = -CLOCKWISE_TARGET;

AccelStepper xStepper(AccelStepper::FULL4WIRE, X_PIN_IN1, X_PIN_IN2, X_PIN_IN3, X_PIN_IN4);

// ================= Serial input =================
const unsigned int MAX_LINE_LENGTH = 64;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);

  xStepper.setMaxSpeed(X_MAX_SPEED);
  xStepper.setAcceleration(X_ACCELERATION);
  xStepper.setCurrentPosition(0);

  Serial.println("Ready. Send X,Y,VX,VY: VX > 0 turns X 15 deg clockwise, VX < 0 counterclockwise");
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
// Reads VX from one "X,Y,VX,VY" line and points the X motor
// clockwise (VX > 0) or counterclockwise (VX < 0).
// ---------------------------------------------------------------
void processLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  float values[4];
  if (!parseFourFloats(line, values)) {
    Serial.println("Invalid input. Expected format: X,Y,VX,VY");
    return;
  }

  float velX = values[2];

  if (velX > 0.0f) {
    xStepper.moveTo(CLOCKWISE_TARGET);
    Serial.println("VX > 0: X clockwise 15 deg");
  } else if (velX < 0.0f) {
    xStepper.moveTo(COUNTERCLOCKWISE_TARGET);
    Serial.println("VX < 0: X counterclockwise 15 deg");
  }
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
