/**
 * esp32_servo_controller.ino
 *
 * Drone Takip Sistemi — ESP32 Pan-Tilt Servo Kontrolcüsü
 *
 * Görev:
 *   UART0 (USB-VCP, 115200 baud) üzerinden Python SerialController'dan
 *   "PAN:<deg>,TILT:<deg>\n" formatında komutlar alır.
 *   İki MG996R servo'ya ESP32 LEDC (PWM) ile hedef açı komutunu uygular.
 *   Ani hareketi yumuşatmak için lineer hızlanma rampalama (acceleration ramp)
 *   kullanılır.
 *
 * Donanım Bağlantıları (plan.txt § 3):
 *   GPIO 18  → Pan  servo  PWM sinyali (turuncu/sarı kablo)
 *   GPIO 19  → Tilt servo  PWM sinyali (turuncu/sarı kablo)
 *   GND      → Ortak GND (servo GND + harici 5V adaptör GND birleşik)
 *   Servo 5V → Harici 5V/5A adaptörden (ESP32 5V pininden BESLEMEYINIZ!)
 *
 * Komut Protokolü:
 *   Gelen: "PAN:45.0,TILT:20.0\n"
 *   Yanıt: "OK PAN:45.0 TILT:20.0\n"   (onay, opsiyonel debug için)
 *
 *   "PING\n"  → "PONG\n"    (T9 bağlantı doğrulama testi için)
 *   "RESET\n" → "RESET OK\n" + servoları merkeze götür
 *
 * Güvenlik:
 *   Açı değerleri [PAN_MIN, PAN_MAX] ve [TILT_MIN, TILT_MAX] aralıklarına kırpılır.
 *   115200 baud → Python tarafındaki SerialController ile eşleşmeli.
 *
 * Derleme Gereksinimleri:
 *   Board: "ESP32 Dev Module" (Arduino IDE → Boards Manager → esp32 by Espressif)
 *   Kütüphane: ESP32Servo (Arduino IDE → Library Manager)
 *              veya Arduino IDE 2.x built-in ESP32 LEDC servo desteği.
 */

#include <ESP32Servo.h>

// ── Pin Tanımları ──────────────────────────────────────────────────────────
static const int PIN_PAN  = 18;
static const int PIN_TILT = 19;

// ── Servo Açı Sınırları ────────────────────────────────────────────────────
// Pan: merkez=90°, -60° → 30°, +60° → 150°
static const float PAN_CENTER   = 90.0f;
static const float PAN_MIN_DEG  = -60.0f;  // yazılım sınırı (derece, merkez=0)
static const float PAN_MAX_DEG  =  60.0f;

// Tilt: merkez=90°, 0° yatay → 90°, 60° yukarı → 150°
static const float TILT_CENTER   = 90.0f;
static const float TILT_MIN_DEG  =  0.0f;
static const float TILT_MAX_DEG  = 60.0f;

// MG996R için PWM microsaniye aralıkları (ölçüm ile kalibre edin)
static const int SERVO_PWM_MIN_US = 500;
static const int SERVO_PWM_MAX_US = 2500;

// ── Rampalama Parametreleri ────────────────────────────────────────────────
// Servo her loop() döngüsünde maksimum bu kadar derece hareket edebilir.
// Düşük değer → daha yumuşak hareket ama yavaş takip.
// Önerilen: 2.0° / loop (50Hz → 100°/sn max hız)
static const float MAX_STEP_DEG = 2.0f;

// ── Durum Değişkenleri ─────────────────────────────────────────────────────
Servo servoPan;
Servo servoTilt;

float targetPan  = PAN_CENTER;   // Hedef açı (servo referansı: 0-180)
float targetTilt = TILT_CENTER;
float currentPan  = PAN_CENTER;
float currentTilt = TILT_CENTER;

String inputBuffer = "";
bool commandReady  = false;

// ── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

/**
 * Bir float değeri [minVal, maxVal] aralığına kırpar.
 */
float clampf(float val, float minVal, float maxVal) {
  if (val < minVal) return minVal;
  if (val > maxVal) return maxVal;
  return val;
}

/**
 * Kullanıcı koordinat sisteminden (merkez=0) servo açısına (merkez=90) dönüştürür.
 * Pan  : kullanıcı [-60, +60] → servo [30, 150]
 * Tilt : kullanıcı [0, +60]   → servo [90, 150]
 */
float panUserToServo(float userDeg) {
  float clamped = clampf(userDeg, PAN_MIN_DEG, PAN_MAX_DEG);
  return PAN_CENTER + clamped;
}

float tiltUserToServo(float userDeg) {
  float clamped = clampf(userDeg, TILT_MIN_DEG, TILT_MAX_DEG);
  return TILT_CENTER + clamped;
}

/**
 * Lineer rampalama: mevcut açıdan hedef açıya MAX_STEP_DEG adımıyla yaklaş.
 */
float ramp(float current, float target, float maxStep) {
  float diff = target - current;
  if (diff >  maxStep) return current + maxStep;
  if (diff < -maxStep) return current - maxStep;
  return target;
}

/**
 * Her iki servoyu mevcut current* değerleriyle günceller.
 */
void applyServos() {
  servoPan.write((int)currentPan);
  servoTilt.write((int)currentTilt);
}

/**
 * Servoları merkez konumuna götürür ve global değişkenleri sıfırlar.
 */
void resetServos() {
  targetPan  = PAN_CENTER;
  targetTilt = TILT_CENTER;
  currentPan  = PAN_CENTER;
  currentTilt = TILT_CENTER;
  applyServos();
  Serial.println("RESET OK");
}

// ── Komut Ayrıştırıcı ─────────────────────────────────────────────────────

/**
 * "PAN:45.0,TILT:20.0" formatındaki komutu ayrıştırır.
 * Başarılı olursa targetPan ve targetTilt güncellenir.
 *
 * Returns: true → komut geçerli; false → format hatası.
 */
bool parseCommand(const String& cmd) {
  // --- PING / RESET özel komutları ---
  if (cmd == "PING") {
    Serial.println("PONG");
    return true;
  }
  if (cmd == "RESET") {
    resetServos();
    return true;
  }

  // --- PAN:x,TILT:y formatı ---
  int panIdx  = cmd.indexOf("PAN:");
  int tiltIdx = cmd.indexOf("TILT:");
  int commaIdx = cmd.indexOf(',');

  if (panIdx < 0 || tiltIdx < 0 || commaIdx < 0) {
    Serial.print("ERR PARSE:");
    Serial.println(cmd);
    return false;
  }

  // PAN değerini çıkar
  String panStr  = cmd.substring(panIdx + 4, commaIdx);
  String tiltStr = cmd.substring(tiltIdx + 5);

  float userPan  = panStr.toFloat();
  float userTilt = tiltStr.toFloat();

  // Servo koordinat sistemine çevir ve kırp
  targetPan  = panUserToServo(userPan);
  targetTilt = tiltUserToServo(userTilt);

  // Onay mesajı (Python tarafında okunmaz ama seri monitörde görmek için yararlı)
  Serial.print("OK PAN:");
  Serial.print(userPan, 1);
  Serial.print(" TILT:");
  Serial.println(userTilt, 1);

  return true;
}

// ── Setup ──────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // ESP32Servo kütüphanesi için timer tahsisi
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);

  servoPan.setPeriodHertz(50);   // Standard 50 Hz servo
  servoTilt.setPeriodHertz(50);

  servoPan.attach(PIN_PAN,   SERVO_PWM_MIN_US, SERVO_PWM_MAX_US);
  servoTilt.attach(PIN_TILT, SERVO_PWM_MIN_US, SERVO_PWM_MAX_US);

  resetServos();

  Serial.println("DRONE_SERVO_CTRL v1.0 READY");
  Serial.print("PAN_PIN:");
  Serial.print(PIN_PAN);
  Serial.print(" TILT_PIN:");
  Serial.println(PIN_TILT);
}

// ── Loop ───────────────────────────────────────────────────────────────────

void loop() {
  // ── 1. Seri veri oku (satır sonuna kadar biriktir) ─────────────────────
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        commandReady = true;
      }
    } else {
      inputBuffer += c;
      // Tampon taşmasına karşı koruma (256 byte)
      if (inputBuffer.length() > 256) {
        inputBuffer = "";
      }
    }
  }

  // ── 2. Komut işle ──────────────────────────────────────────────────────
  if (commandReady) {
    inputBuffer.trim();
    parseCommand(inputBuffer);
    inputBuffer  = "";
    commandReady = false;
  }

  // ── 3. Rampalama ile servoları güncelle ────────────────────────────────
  currentPan  = ramp(currentPan,  targetPan,  MAX_STEP_DEG);
  currentTilt = ramp(currentTilt, targetTilt, MAX_STEP_DEG);
  applyServos();

  // 50 Hz servo güncelleme hızı → 20ms bekleme
  delay(20);
}
