// Receives ball data from demo.py over Wi-Fi (UDP).
// Each packet: t,valid,x,y,vx,vy   e.g. "7.14,1,-8.58,10.65,-1.37,9.13"
//   valid 1 = position known, 0 = no data (hold or stop the motors)
//   x, y   metres from the centre spot; vx, vy m/s, constant for each pass
//
// The ESP32 makes its own Wi-Fi network (works anywhere, no venue Wi-Fi needed):
//   1. Upload, then join the network "HapticBall" (password below) from the Mac.
//   2. On the Mac run:  python demo.py VIDEO --udp 192.168.4.1:5005
#include <WiFi.h>
#include <WiFiUdp.h>

const char* NETWORK = "HapticBall";
const char* PASSWORD = "soccer2026";   // at least 8 characters
const int PORT = 5005;

WiFiUDP udp;
char packet[96];

void onBallData(float t, bool valid, float x, float y, float vx, float vy) {
  // TODO: drive the motors here. For now, echo what arrived.
  Serial.printf("t=%.2f valid=%d x=%.1f y=%.1f vx=%.1f vy=%.1f\n", t, valid, x, y, vx, vy);
}

void setup() {
  Serial.begin(115200);
  WiFi.softAP(NETWORK, PASSWORD);
  Serial.printf("Wi-Fi \"%s\" ready; send to %s:%d\n", NETWORK, WiFi.softAPIP().toString().c_str(), PORT);
  udp.begin(PORT);
}

void loop() {
  int size = udp.parsePacket();
  if (size > 0) {
    int n = udp.read(packet, sizeof(packet) - 1);
    packet[n] = '\0';
    float t, x, y, vx, vy;
    int valid;
    if (sscanf(packet, "%f,%d,%f,%f,%f,%f", &t, &valid, &x, &y, &vx, &vy) == 6) {
      onBallData(t, valid == 1, x, y, vx, vy);
    }
  }
}
