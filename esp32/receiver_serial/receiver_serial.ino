// Receives ball data from demo.py over USB serial (or Bluetooth serial, see BLUETOOTH below).
// Each line: t,valid,x,y,vx,vy   e.g. "7.14,1,-8.58,10.65,-1.37,9.13"
//   valid 1 = position known, 0 = no data (hold or stop the motors)
//   x, y   metres from the centre spot (+x toward the right-hand goal, +y toward the near touchline)
//   vx, vy m/s, constant for each pass
//
// USB: upload, then on the Mac run  python demo.py VIDEO --serial /dev/cu.usbserial-XXXX
// BLUETOOTH: uncomment the three BLUETOOTH lines, upload, pair "HapticBall" in the Mac's
// Bluetooth settings, then use  --serial /dev/cu.HapticBall

// #include "BluetoothSerial.h"   // BLUETOOTH
// BluetoothSerial SerialBT;      // BLUETOOTH
#define LINK Serial               // BLUETOOTH: change to SerialBT

char line[96];
size_t len = 0;

void onBallData(float t, bool valid, float x, float y, float vx, float vy) {
  // TODO: drive the motors here. For now, echo what arrived.
  Serial.printf("t=%.2f valid=%d x=%.1f y=%.1f vx=%.1f vy=%.1f\n", t, valid, x, y, vx, vy);
}

void setup() {
  Serial.begin(115200);
  // SerialBT.begin("HapticBall");  // BLUETOOTH
}

void loop() {
  while (LINK.available()) {
    char c = LINK.read();
    if (c == '\n') {
      line[len] = '\0';
      float t, x, y, vx, vy;
      int valid;
      if (sscanf(line, "%f,%d,%f,%f,%f,%f", &t, &valid, &x, &y, &vx, &vy) == 6) {
        onBallData(t, valid == 1, x, y, vx, vy);
      }
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}
