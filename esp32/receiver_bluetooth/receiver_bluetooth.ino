// Receives ball data from demo.py over Bluetooth (classic Bluetooth serial).
// Needs an original ESP32 (ESP32-WROOM / ESP32-DevKit); ESP32-S3/C3/C6 have no classic Bluetooth.
//
// Each line: t,valid,x,y,vx,vy   e.g. "7.14,1,-8.58,10.65,-1.37,9.13"
//   valid 1 = position known, 0 = no data (hold or stop the motors)
//   x, y   metres from the centre spot (+x toward the right-hand goal, +y toward the near touchline)
//   vx, vy m/s, constant for each pass
//
// 1. Upload. 2. Pair "HapticBall" in the Mac's Bluetooth settings.
// 3. On the Mac:  python demo.py VIDEO --serial /dev/cu.HapticBall
// Everything received is also printed on USB, so the Arduino Serial Monitor (115200) shows it.
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;
char line[96];
size_t len = 0;

void onBallData(float t, bool valid, float x, float y, float vx, float vy) {
  // TODO: drive the motors here.
  Serial.printf("t=%.2f valid=%d x=%.1f y=%.1f vx=%.1f vy=%.1f\n", t, valid, x, y, vx, vy);
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("HapticBall");
  Serial.println("Bluetooth \"HapticBall\" ready to pair");
}

void loop() {
  while (SerialBT.available()) {
    char c = SerialBT.read();
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
