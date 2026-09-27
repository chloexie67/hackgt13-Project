/*
  Ball Position + Velocity -> Stepper Target Position
  ------------------------------------------------------
  Reads live ball data over USB Serial (115200 baud), one line at a time, in the form:
      X,Y,VX,VY\n
  e.g.  "12.3,-4.8,1.5,-0.2\n"   (no spaces, comma separated)

  demo.py sends these lines about 20 times a second while it plays the match clip.
  X, Y are metres from the centre spot; VX, VY are m/s and stay the same for the
  whole of each pass, so a change in velocity means a new pass. No lines are sent
  while there's no ball data, so the steppers hold their last position.
  Close the Arduino Serial Monitor before running demo.py: only one program can
  use the USB port at a time.

  STATE MACHINE
  --------------
  TRACKING:
    Normal state. X and Y position are mapped linearly to a tilt
    angle, and that angle is converted to a stepper step position
    (centered at 0). This continuously reflects where the ball is
    on the field.

  KICKED:
    Triggered when a new pass starts (the velocity changes) and its
    speed exceeds a defined threshold (i.e. the ball was just kicked). Instead of tracking
    position, the steppers perform a sudden "jerk": for each axis,
    the stepper steps repeatedly, at a fixed step rate, for a fixed
    number of ticks, in the direction of that axis's velocity
    component. Positive velocity (X or Y) = clockwise (CW).
    Negative velocity (X or Y) = counter-clockwise (CCW).
    Steps never go past the tilt angle limits.
    Once both axes finish their ticks, the state returns to TRACKING.

  NOTE: This script only computes and prints the TARGET STEP POSITION
  for the X and Y steppers (plus current state). It does not contain
  any motor/driver code (no step()/direction pin toggling, no
  AccelStepper, etc.) — that will be added separately once this logic
  is confirmed correct.
*/

// ================= Field geometry =================
const float X_FIELD_MAX = 105.0f / 2.0f;   // +/- 52.5 m
const float Y_FIELD_MAX = 68.0f  / 2.0f;   // +/- 34 m (the tracker's pitch is 105 x 68 m)

// ================= Tilt angle limits =================
const float X_ANGLE_MAX_DEG = 20.0f;       // +/- 20 degrees
const float Y_ANGLE_MAX_DEG = 15.0f;       // +/- 15 degrees

// ================= Stepper motor characteristics =================
const int   X_STEPS_PER_REV = 200;                       // given
const float X_DEG_PER_STEP  = 360.0f / X_STEPS_PER_REV;  // = 1.8 deg/step

const int   Y_STEPS_PER_REV = 2048;                      // given
const float Y_DEG_PER_STEP  = 360.0f / Y_STEPS_PER_REV;  // = ~0.176 deg/step

// Furthest each stepper may go from centre (the tilt angle limits, in steps)
const long X_STEP_LIMIT = (long) (X_ANGLE_MAX_DEG / X_DEG_PER_STEP);   // 11 steps
const long Y_STEP_LIMIT = (long) (Y_ANGLE_MAX_DEG / Y_DEG_PER_STEP);   // 85 steps

// ================= Kick ("jerk") configuration =================
const float VELOCITY_KICK_THRESHOLD = 2.0f;    // m/s -- a new pass faster than this = "kicked"
const float NEW_PASS_VELOCITY_CHANGE = 0.05f;  // m/s -- velocity changed by more than this = new pass
const int   KICK_STEP_RATE_HZ       = 100;     // steps per second during a kick
const int   KICK_TOTAL_TICKS        = 20;      // number of steps to jerk through
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

// ================= Serial input buffer =================
const unsigned int MAX_LINE_LENGTH = 64;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  Serial.println("Ready. Send data as X,Y,VX,VY (e.g. 10.5,-3.2,1.8,-0.4)");
}

void loop() {
  // Read serial data one line at a time
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

  // If a kick is in progress, keep advancing it every loop iteration
  if (currentState == KICKED) {
    updateKick();
  }
}

// ---------------------------------------------------------------
// Parses one line of "X,Y,VX,VY" and either updates tracking
// position or starts a kick, depending on whether a new, fast pass
// has started.
// ---------------------------------------------------------------
void processLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  float values[4];
  if (!parseFourFloats(line, values)) {
    Serial.println("Invalid input. Expected format: X,Y,VX,VY");
    return;
  }

  float ballX  = values[0];
  float ballY  = values[1];
  float velX   = values[2];
  float velY   = values[3];

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
      token = line.substring(startIndex); // last value, no trailing comma
    }

    token.trim();
    if (token.length() == 0) return false;
    outValues[i] = token.toFloat();
  }
  return true;
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
    xStepPosition = constrain(xStepPosition + xKickDirection, -X_STEP_LIMIT, X_STEP_LIMIT); // +1 = CW, -1 = CCW
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

  positionValue : ball coordinate from demo.py (e.g. ballX or ballY)
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
