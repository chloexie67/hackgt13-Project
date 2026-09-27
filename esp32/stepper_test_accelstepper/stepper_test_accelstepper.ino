/*
  Stepper test with AccelStepper
  ------------------------------
  Tests both tilt motors on their own, then together, using the same pins,
  step counts and tilt limits as ball_tilt_position.

  Needs the "AccelStepper" library by Mike McCauley (Tools -> Manage Libraries).

  Level the board, power it up, upload, then open the Serial Monitor at 115200
  (line ending: Newline). Type a command and press Enter:
    x  test the X motor (along the pitch, 200 steps/rev, pins 32, 33, 25, 26)
    y  test the Y motor (across the pitch, 2048 steps/rev, pins 27, 12, 14, 13)
    b  both motors at once
    k  kick: a quick jerk out and back on both motors
    a  all of the above
    0  return both motors to centre
    -  halve both motors' speed and acceleration
    +  double both motors' speed and acceleration

  Each move prints what the board should do. PASS means AccelStepper reached the
  target within the expected time; steppers have no position sensor, so check
  by eye that the board actually moved that way and came back level. If a motor
  buzzes without turning, its coil order is wrong (see the pin notes below).

  SHAKING WITHOUT TURNING: press - a few times and test again. If the motor
  turns at a lower speed, that speed was too high; put the working values into
  the main sketch. If it still only shakes at the slowest speed, it isn't speed:
  swap the middle two pins on that motor's AccelStepper line (coil order), and
  check the motor supply and that the driver shares ground with the ESP32.
*/

#include <AccelStepper.h>

// ================= Pins (same as ball_tilt_position) =================
const int X_PIN_IN1 = 32;
const int X_PIN_IN2 = 33;
const int X_PIN_IN3 = 25;
const int X_PIN_IN4 = 26;

const int Y_PIN_IN1 = 27;
const int Y_PIN_IN2 = 12;   // GPIO12 must be low at boot, or the ESP32 won't start
const int Y_PIN_IN3 = 14;
const int Y_PIN_IN4 = 13;

// ================= Motors and tilt limits =================
const int   X_STEPS_PER_REV = 200;
const int   Y_STEPS_PER_REV = 2048;
const float X_ANGLE_MAX_DEG = 20.0f;
const float Y_ANGLE_MAX_DEG = 15.0f;
const long  X_STEP_LIMIT = (long) (X_ANGLE_MAX_DEG / (360.0f / X_STEPS_PER_REV));   // 11 steps
const long  Y_STEP_LIMIT = (long) (Y_ANGLE_MAX_DEG / (360.0f / Y_STEPS_PER_REV));   // 85 steps

// Starting speeds, deliberately slow; change them live with - and +
float xMaxSpeed    = 100.0f;   // steps/s (30 RPM)
float xAcceleration = 200.0f;  // steps/s^2
float yMaxSpeed    = 200.0f;   // steps/s (~6 RPM)
float yAcceleration = 400.0f;  // steps/s^2
const float MIN_MAX_SPEED = 5.0f;

const unsigned long PAUSE_MS = 500;     // pause between moves so each one is easy to see

// FULL4WIRE takes the coils in firing order. For an H-bridge that's
// IN1, IN2, IN3, IN4; a 28BYJ-48 on a ULN2003 fires IN1, IN3, IN2, IN4.
AccelStepper xStepper(AccelStepper::FULL4WIRE, X_PIN_IN1, X_PIN_IN2, X_PIN_IN3, X_PIN_IN4);
AccelStepper yStepper(AccelStepper::FULL4WIRE, Y_PIN_IN1, Y_PIN_IN3, Y_PIN_IN2, Y_PIN_IN4);

int testsPassed = 0;
int testsFailed = 0;

void setup() {
  Serial.begin(115200);

  applySpeeds();
  xStepper.setCurrentPosition(0);
  yStepper.setCurrentPosition(0);

  delay(500);
  Serial.println();
  Serial.println("Stepper test (AccelStepper). Board should be level now.");
  Serial.print("X limit: +/-"); Serial.print(X_STEP_LIMIT); Serial.println(" steps (20 deg)");
  Serial.print("Y limit: +/-"); Serial.print(Y_STEP_LIMIT); Serial.println(" steps (15 deg)");
  printSpeeds();
  printHelp();
}

void loop() {
  if (Serial.available() == 0) return;

  char command = Serial.read();
  if (command == '\n' || command == '\r' || command == ' ') return;

  testsPassed = 0;
  testsFailed = 0;

  switch (command) {
    case 'x': testXMotor(); break;
    case 'y': testYMotor(); break;
    case 'b': testBothMotors(); break;
    case 'k': testKick(); break;
    case 'a': testXMotor(); testYMotor(); testBothMotors(); testKick(); break;
    case '0': moveAndCheck("Return to centre", 0, 0, "board level"); break;
    case '-':
    case '+': {
      float factor = (command == '+') ? 2.0f : 0.5f;
      if (command == '-' && xMaxSpeed * factor < MIN_MAX_SPEED) {
        Serial.println("Already at the slowest speed.");
      } else {
        xMaxSpeed *= factor; xAcceleration *= factor;
        yMaxSpeed *= factor; yAcceleration *= factor;
        applySpeeds();
      }
      printSpeeds();
      return;
    }
    default:
      Serial.print("Unknown command: ");
      Serial.println(command);
      printHelp();
      return;
  }

  Serial.print("Done: ");
  Serial.print(testsPassed);
  Serial.print(" passed, ");
  Serial.print(testsFailed);
  Serial.println(" failed");
  printHelp();
}

void printHelp() {
  Serial.println("Commands: x = X motor, y = Y motor, b = both, k = kick, a = all, 0 = centre, - slower, + faster");
}

void applySpeeds() {
  xStepper.setMaxSpeed(xMaxSpeed);
  xStepper.setAcceleration(xAcceleration);
  yStepper.setMaxSpeed(yMaxSpeed);
  yStepper.setAcceleration(yAcceleration);
}

void printSpeeds() {
  Serial.print("X: "); Serial.print(xMaxSpeed); Serial.print(" steps/s (");
  Serial.print(xMaxSpeed * 60.0f / X_STEPS_PER_REV); Serial.print(" RPM), accel "); Serial.println(xAcceleration);
  Serial.print("Y: "); Serial.print(yMaxSpeed); Serial.print(" steps/s (");
  Serial.print(yMaxSpeed * 60.0f / Y_STEPS_PER_REV); Serial.print(" RPM), accel "); Serial.println(yAcceleration);
}

// ---------------------------------------------------------------
// X alone: full tilt one way, back to centre, full tilt the other
// way, back to centre, then a small move to check fine steps.
// ---------------------------------------------------------------
void testXMotor() {
  Serial.println("--- X motor ---");
  moveAndCheck("X to +limit", X_STEP_LIMIT, 0, "tilts fully toward one goal");
  moveAndCheck("X to centre", 0, 0, "level again");
  moveAndCheck("X to -limit", -X_STEP_LIMIT, 0, "tilts fully toward the other goal");
  moveAndCheck("X to centre", 0, 0, "level again");
  moveAndCheck("X small move (+2 steps)", 2, 0, "tilts slightly (3.6 deg)");
  moveAndCheck("X to centre", 0, 0, "level again");
}

// ---------------------------------------------------------------
// Y alone: the same pattern across the pitch.
// ---------------------------------------------------------------
void testYMotor() {
  Serial.println("--- Y motor ---");
  moveAndCheck("Y to +limit", 0, Y_STEP_LIMIT, "tilts fully toward one touchline");
  moveAndCheck("Y to centre", 0, 0, "level again");
  moveAndCheck("Y to -limit", 0, -Y_STEP_LIMIT, "tilts fully toward the other touchline");
  moveAndCheck("Y to centre", 0, 0, "level again");
  moveAndCheck("Y small move (+10 steps)", 0, 10, "tilts slightly (1.8 deg)");
  moveAndCheck("Y to centre", 0, 0, "level again");
}

// ---------------------------------------------------------------
// Both at once, corner to corner, to check neither motor slows or
// blocks the other.
// ---------------------------------------------------------------
void testBothMotors() {
  Serial.println("--- Both motors ---");
  moveAndCheck("Both to (+X, +Y) corner", X_STEP_LIMIT, Y_STEP_LIMIT, "tilts toward one corner");
  moveAndCheck("Both to (-X, -Y) corner", -X_STEP_LIMIT, -Y_STEP_LIMIT, "tilts toward the opposite corner");
  moveAndCheck("Both to (+X, -Y) corner", X_STEP_LIMIT, -Y_STEP_LIMIT, "tilts toward a third corner");
  moveAndCheck("Both to centre", 0, 0, "level again");
}

// ---------------------------------------------------------------
// A kick like ball_tilt_position's: a quick jerk in one direction,
// then straight back, with no pause in between.
// ---------------------------------------------------------------
void testKick() {
  Serial.println("--- Kick ---");
  xStepper.setAcceleration(xAcceleration * 4);
  yStepper.setAcceleration(yAcceleration * 4);
  moveAndCheck("Kick toward (+X, +Y)", X_STEP_LIMIT, 20, "sharp jolt toward one corner");
  moveAndCheck("Back to centre", 0, 0, "level again");
  moveAndCheck("Kick toward (-X, -Y)", -X_STEP_LIMIT, -20, "sharp jolt the other way");
  moveAndCheck("Back to centre", 0, 0, "level again");
  applySpeeds();
}

// ---------------------------------------------------------------
// Moves both motors to the given targets and checks they get there
// within the time AccelStepper should need (plus a margin).
// ---------------------------------------------------------------
void moveAndCheck(const char *name, long xTarget, long yTarget, const char *expected) {
  long xDistance = labs(xTarget - xStepper.currentPosition());
  long yDistance = labs(yTarget - yStepper.currentPosition());
  unsigned long timeLimitMs = 500 + max(expectedMoveMs(xDistance, xMaxSpeed, xAcceleration),
                                        expectedMoveMs(yDistance, yMaxSpeed, yAcceleration)) * 2;

  Serial.print(name);
  Serial.print(" -> should see: ");
  Serial.println(expected);

  xStepper.moveTo(xTarget);
  yStepper.moveTo(yTarget);
  unsigned long startMs = millis();
  while ((xStepper.distanceToGo() != 0 || yStepper.distanceToGo() != 0) &&
         millis() - startMs < timeLimitMs) {
    xStepper.run();
    yStepper.run();
  }
  unsigned long elapsedMs = millis() - startMs;

  bool reached = xStepper.currentPosition() == xTarget && yStepper.currentPosition() == yTarget;
  Serial.print(reached ? "  PASS" : "  FAIL");
  Serial.print(" in ");
  Serial.print(elapsedMs);
  Serial.print(" ms (limit ");
  Serial.print(timeLimitMs);
  Serial.print(" ms), X at ");
  Serial.print(xStepper.currentPosition());
  Serial.print(", Y at ");
  Serial.println(yStepper.currentPosition());
  if (reached) testsPassed++; else testsFailed++;

  delay(PAUSE_MS);
}

// ---------------------------------------------------------------
// Time for a trapezoidal move (accelerate, cruise, decelerate) of
// the given number of steps, in ms.
// ---------------------------------------------------------------
unsigned long expectedMoveMs(long steps, float maxSpeed, float acceleration) {
  if (steps == 0) return 0;
  float rampSteps = maxSpeed * maxSpeed / (2.0f * acceleration);
  float seconds;
  if (steps < 2 * rampSteps) {
    seconds = 2.0f * sqrt(steps / acceleration);   // never reaches full speed
  } else {
    seconds = 2.0f * maxSpeed / acceleration + (steps - 2 * rampSteps) / maxSpeed;
  }
  return (unsigned long) (seconds * 1000.0f);
}
