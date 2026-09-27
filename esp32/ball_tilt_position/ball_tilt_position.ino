/*
  Ball Position + Velocity -> Stepper Target Position
  ------------------------------------------------------
  Receives ball data from demo.py over Bluetooth, one line at a time:
      t,valid,x,y,vx,vy\n
  e.g.  "7.14,1,-8.58,10.65,-1.37,9.13\n"
    valid 1 = ball position known; 0 = no data (replay, ball lost or in the air)
    x, y   metres from the centre spot (+x toward the right-hand goal, +y toward the near touchline)
    vx, vy m/s; demo.py sends ONE constant velocity per pass, so a change of velocity = a new pass
  For testing by hand, the Serial Monitor (USB, 115200) also accepts  X,Y,VX,VY

  Bluetooth: needs an original ESP32 (ESP32-WROOM / DevKit). Upload, pair "HapticBall" on the
  Mac, then run:  python demo.py VIDEO --serial /dev/cu.HapticBall

  STATE MACHINE
  --------------
  TRACKING:
    Normal state. X and Y position are mapped linearly to a tilt
    angle, and that angle is converted to a stepper step position
    (centered at 0). This continuously reflects where the ball is
    on the field. Lines with valid=0 are ignored: the steppers hold.

  KICKED:
    Triggered when a NEW pass starts (the velocity changes) and its speed
    exceeds a defined threshold (i.e. the ball was just kicked). Instead of tracking
    position, the steppers perform a sudden "jerk": for each axis,
    the stepper steps repeatedly, at a fixed step rate, for a fixed
    number of ticks, in the direction of that axis's velocity
    component. Positive velocity (X or Y) = clockwise (CW).
    Negative velocity (X or Y) = counter-clockwise (CCW).
    Steps never go past the tilt limits.
    Once both axes finish their ticks, the state returns to TRACKING.

  NOTE: This script only computes and prints the TARGET STEP POSITION
  for the X and Y steppers (plus current state). It does not contain
  any motor/driver code (no step()/direction pin toggling, no
  AccelStepper, etc.) — that will be added separately once this logic
  is confirmed correct.
*/
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// ================= Field geometry (matches the tracker's 105 x 68 m pitch) =================
const float X_FIELD_MAX = 105.0f / 2.0f;   // +/- 52.5 m
const float Y_FIELD_MAX = 68.0f  / 2.0f;   // +/- 34 m

// ================= Tilt angle limits =================
const float X_ANGLE_MAX_DEG = 20.0f;       // +/- 20 degrees
const float Y_ANGLE_MAX_DEG = 15.0f;       // +/- 15 degrees

// ================= Stepper motor characteristics =================
const int   X_STEPS_PER_REV = 200;                       // given
const float X_DEG_PER_STEP  = 360.0f / X_STEPS_PER_REV;  // = 1.8 deg/step

const int   Y_STEPS_PER_REV = 2048;                      // given
const float Y_DEG_PER_STEP  = 360.0f / Y_STEPS_PER_REV;  // = ~0.176 deg/step

// Furthest each stepper may go from centre (the tilt limits, in steps)
const long X_STEP_LIMIT = (long) (X_ANGLE_MAX_DEG / X_DEG_PER_STEP);
const long Y_STEP_LIMIT = (long) (Y_ANGLE_MAX_DEG / Y_DEG_PER_STEP);

// ================= Kick ("jerk") configuration =================
const float VELOCITY_KICK_THRESHOLD = 2.0f;    // m/s -- a new pass faster than this = "kicked"
const float NEW_PASS_VELOCITY_CHANGE = 0.05f;  // m/s -- velocity changed by more than this = new pass
const int   KICK_STEP_RATE_HZ       = 100;     // steps per second during a kick
const int   KICK_TOTAL_TICKS        = 100;     // number of steps to jerk through (stops at the tilt limit)
const unsigned long KICK_STEP_INTERVAL_MS = 1000UL / KICK_STEP_RATE_HZ;

// ================= State machine =================
enum BallState { TRACKING, KICKED };
BallState currentState = TRACKING;

// Persistent stepper positions (centered at 0, carried between states)
long xStepPosition = 0;
long yStepPosition = 0;

// Most recently received ball position (used to resume TRACKING after a kick)
float lastBallX = 0.0f;
float lastBallY = 0.0f;

// Velocity of the current pass (a change means a new pass)
float lastVelX = 0.0f;
float lastVelY = 0.0f;

// Kick bookkeeping (per axis, independent direction/tick tracking)
int xKickDirection = 0;      // +1 = CW, -1 = CCW, 0 = no motion this axis
int yKickDirection = 0;
int xTicksRemaining = 0;
int yTicksRemaining = 0;
unsigned long xLastStepTime = 0;
unsigned long yLastStepTime = 0;

// ================= Input buffers (Bluetooth from demo.py, USB for testing) =================
const unsigned int MAX_LINE_LENGTH = 96;
String bluetoothBuffer = "";
String usbBuffer = "";

void setup() {
  Serial.begin(115200);
  SerialBT.begin("HapticBall");
  Serial.println("Ready. Bluetooth \"HapticBall\" waiting for demo.py; USB accepts X,Y,VX,VY for testing");
}

void loop() {
  // Read data one line at a time from Bluetooth (demo.py) and USB (manual tests)
  readLines(SerialBT, bluetoothBuffer);
  readLines(Serial, usbBuffer);

  // If a kick is in progress, keep advancing it every loop iteration
  if (currentState == KICKED) {
    updateKick();
  }
}

void readLines(Stream &input, String &buffer) {
  while (input.available() > 0) {
    char incomingChar = input.read();

    if (incomingChar == '\n') {
      processLine(buffer);
      buffer = "";
    } else if (incomingChar != '\r') {
      if (buffer.length() < MAX_LINE_LENGTH) {
        buffer += incomingChar;
      } else {
        buffer = "";  // garbage without a newline: drop it
      }
    }
  }
}

// ---------------------------------------------------------------
// Parses one line ("t,valid,x,y,vx,vy" from demo.py, or "X,Y,VX,VY"
// typed by hand) and either updates tracking position or starts a
// kick, depending on whether a new, fast pass has started.
// ---------------------------------------------------------------
void processLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  float values[6];
  int count = parseFloats(line, values, 6);
  float ballX, ballY, velX, velY;
  if (count == 6) {
    if (values[1] < 0.5f) return;   // valid = 0: no data, hold the current position
    ballX = values[2]; ballY = values[3]; velX = values[4]; velY = values[5];
  } else if (count == 4) {
    ballX = values[0]; ballY = values[1]; velX = values[2]; velY = values[3];
  } else {
    Serial.println("Invalid input. Expected t,valid,x,y,vx,vy or X,Y,VX,VY");
    return;
  }

  lastBallX = ballX;
  lastBallY = ballY;

  bool newPass = fabs(velX - lastVelX) > NEW_PASS_VELOCITY_CHANGE ||
                 fabs(velY - lastVelY) > NEW_PASS_VELOCITY_CHANGE;
  lastVelX = velX;
  lastVelY = velY;

  // Ignore new lines while a kick is still playing out
  if (currentState == KICKED) {
    return;
  }

  float velocityMagnitude = sqrt(velX * velX + velY * velY);

  if (newPass && velocityMagnitude > VELOCITY_KICK_THRESHOLD) {
    startKick(velX, velY);
  } else {
    xStepPosition = computeStepPosition(ballX, X_FIELD_MAX, X_ANGLE_MAX_DEG, X_DEG_PER_STEP);
    yStepPosition = computeStepPosition(ballY, Y_FIELD_MAX, Y_ANGLE_MAX_DEG, Y_DEG_PER_STEP);
    printStatus();
  }
}

// ---------------------------------------------------------------
// Splits "a,b,c,..." into up to maxValues floats. Returns how many
// values were read, or 0 on malformed input.
// ---------------------------------------------------------------
int parseFloats(String line, float outValues[], int maxValues) {
  int count = 0;
  int startIndex = 0;
  while (count < maxValues) {
    int commaIndex = line.indexOf(',', startIndex);
    String token = (commaIndex == -1) ? line.substring(startIndex) : line.substring(startIndex, commaIndex);
    token.trim();
    if (token.length() == 0) return 0;
    outValues[count++] = token.toFloat();
    if (commaIndex == -1) return count;
    startIndex = commaIndex + 1;
  }
  return 0;  // more values than expected
}

// ---------------------------------------------------------------
// Begins a kick: sets direction per axis from velocity sign
// (positive = CW, negative = CCW, ~0 = that axis stays still),
// and arms each axis with the configured number of ticks.
// ---------------------------------------------------------------
void startKick(float velX, float velY) {
  currentState = KICKED;

  xKickDirection = (velX > 0.0f) ? 1 : ((velX < 0.0f) ? -1 : 0);
  yKickDirection = (velY > 0.0f) ? 1 : ((velY < 0.0f) ? -1 : 0);

  xTicksRemaining = (xKickDirection != 0) ? KICK_TOTAL_TICKS : 0;
  yTicksRemaining = (yKickDirection != 0) ? KICK_TOTAL_TICKS : 0;

  unsigned long now = millis();
  xLastStepTime = now;
  yLastStepTime = now;

  Serial.println("STATE:KICKED_START");
}

// ---------------------------------------------------------------
// Advances the kick jerk over time (non-blocking), one axis at a
// time, at KICK_STEP_RATE_HZ, until each axis's ticks run out.
// Once both axes are done, returns to TRACKING using the last
// known ball position.
// ---------------------------------------------------------------
void updateKick() {
  unsigned long now = millis();
  bool steppedThisLoop = false;

  if (xTicksRemaining > 0 && (now - xLastStepTime) >= KICK_STEP_INTERVAL_MS) {
    xStepPosition = constrain(xStepPosition + xKickDirection, -X_STEP_LIMIT, X_STEP_LIMIT);
    xTicksRemaining--;
    xLastStepTime = now;
    steppedThisLoop = true;
  }

  if (yTicksRemaining > 0 && (now - yLastStepTime) >= KICK_STEP_INTERVAL_MS) {
    yStepPosition = constrain(yStepPosition + yKickDirection, -Y_STEP_LIMIT, Y_STEP_LIMIT);
    yTicksRemaining--;
    yLastStepTime = now;
    steppedThisLoop = true;
  }

  if (steppedThisLoop) {
    printStatus();
  }

  // Kick finished on both axes -> resume tracking
  if (xTicksRemaining == 0 && yTicksRemaining == 0) {
    currentState = TRACKING;
    xStepPosition = computeStepPosition(lastBallX, X_FIELD_MAX, X_ANGLE_MAX_DEG, X_DEG_PER_STEP);
    yStepPosition = computeStepPosition(lastBallY, Y_FIELD_MAX, Y_ANGLE_MAX_DEG, Y_DEG_PER_STEP);
    Serial.println("STATE:KICKED_END");
    printStatus();
  }
}

// ---------------------------------------------------------------
// Prints the current state and both stepper target positions.
// ---------------------------------------------------------------
void printStatus() {
  Serial.print("STATE:");
  Serial.print(currentState == TRACKING ? "TRACKING" : "KICKED");
  Serial.print(",X_STEP_POSITION:");
  Serial.print(xStepPosition);
  Serial.print(",Y_STEP_POSITION:");
  Serial.println(yStepPosition);
}

/*
  Maps a raw position value to a target stepper step count.

  positionValue : ball coordinate from the tracker (e.g. ballX or ballY)
  positionMax   : max absolute field coordinate for this axis (e.g. 52.5)
  angleMaxDeg   : max absolute tilt angle for this axis (e.g. 20.0)
  degPerStep    : stepper resolution (degrees per step) for this axis

  Returns the target step count relative to center (0 = centered).
*/
long computeStepPosition(float positionValue, float positionMax,
                          float angleMaxDeg, float degPerStep) {
  // 1. Clamp position to valid field range
  if (positionValue > positionMax)  positionValue = positionMax;
  if (positionValue < -positionMax) positionValue = -positionMax;

  // 2. Linearly map position -> tilt angle (centered at 0)
  float angleDeg = (positionValue / positionMax) * angleMaxDeg;

  // 3. Convert angle -> step count (centered at 0)
  long stepPosition = (long) round(angleDeg / degPerStep);

  return stepPosition;
}
