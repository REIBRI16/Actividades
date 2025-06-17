#include "DualVNH5019MotorShield.h"
DualVNH5019MotorShield md;

// Definición de PINs
#define encoder0PinA  19
#define encoder0PinB  18
#define encoder1PinA  20
#define encoder1PinB  21

// Variables Tiempo
unsigned long time_ant = 0;
const int Period = 10000;   // 10 ms = 100Hz
const float dt = Period *0.000001f;

char endChar = ';';               // Carácter que indica el final del mensaje
String incomingMessage;           // Variable donde se almacenará el mensaje recibido
bool messageComplete = false;     // Bandera que indica si el mensaje está completo


float RPM_DESEADA = 70;

float Kp_motor0 = 0;
float Ki_motor0 = 0;
float Kd_motor0 = 0;

float Kp_motor1 = 0;
float Ki_motor1 = 0;
float Kd_motor1 = 0;

float k0_motor0 = 0;
float k1_motor0 = 0;
float k2_motor0 = 0;

float k0_motor1 = 0;
float k1_motor1 = 0;
float k2_motor1 = 0;

float error_actual_motor0 = 0.0;
float error_actual_motor1 = 0.0;

float error_t1_motor0 = 0.0;
float error_t1_motor1 = 0.0;

float error_t2_motor0 = 0.0;
float error_t2_motor1 = 0.0;

float c_motor0 = 0.0;
float c_motor1 = 0.0;

float c_t1_motor0 = 0.0;
float c_t1_motor1 = 0.0;

float motorout0 = 0.0;
float motorout1 = 0.0;


volatile long encoder0Pos = 0;
volatile long encoder1Pos = 0;
long newposition0;
long oldposition0 = 0;
long newposition1;
long oldposition1 = 0;
unsigned long newtime;
float vel0;
float vel1;
int motorSpeedIzq = 0;               // Velocidad del motor
int motorSpeedDer = 0;

int pos_b =-1;

void doEncoder0A()
{
  if (digitalRead(encoder0PinA) == digitalRead(encoder0PinB)) {
    encoder0Pos++;
  } else {
    encoder0Pos--;
  }
}

void doEncoder0B()
{
  if (digitalRead(encoder0PinA) == digitalRead(encoder0PinB)) {
    encoder0Pos--;
  } else {
    encoder0Pos++;
  }
}

void doEncoder1A()
{
  if (digitalRead(encoder1PinA) == digitalRead(encoder1PinB)) {
    encoder1Pos++;
  } else {
    encoder1Pos--;
  }
}

void doEncoder1B()
{
  if (digitalRead(encoder1PinA) == digitalRead(encoder1PinB)) {
    encoder1Pos--;
  } else {
    encoder1Pos++;
  }
}

void setup()
{
    // Configurar MotorShield
  md.init();

  // Configurar Encoders
  pinMode(encoder0PinA, INPUT);
  digitalWrite(encoder0PinA, HIGH);       // Incluir una resistencia de pullup en le entrada
  pinMode(encoder0PinB, INPUT);
  digitalWrite(encoder0PinB, HIGH);       // Incluir una resistencia de pullup en le entrada
  pinMode(encoder1PinA, INPUT);
  digitalWrite(encoder1PinA, HIGH);       // Incluir una resistencia de pullup en le entrada
  pinMode(encoder1PinB, INPUT);
  digitalWrite(encoder1PinB, HIGH);       // Incluir una resistencia de pullup en le entrada
  attachInterrupt(digitalPinToInterrupt(encoder0PinA), doEncoder0A, CHANGE);  // encoder 0 PIN A
  attachInterrupt(digitalPinToInterrupt(encoder0PinB), doEncoder0B, CHANGE);  // encoder 0 PIN B
  attachInterrupt(digitalPinToInterrupt(encoder1PinA), doEncoder1A, CHANGE);  // encoder 1 PIN A
  attachInterrupt(digitalPinToInterrupt(encoder1PinB), doEncoder1B, CHANGE);  // encoder 1 PIN B

  Serial.begin(115200);          // Monitor serial
  Serial3.begin(38400);          // Comunicación Bluetooth
}

void loop()
{
  // Verificamos si hay datos disponibles desde Bluetooth
  if (Serial3.available() > 0) {
    incomingMessage = readBluetoothMessage();  // Leer mensaje entrante

    // Si el mensaje empieza con 'a' y está completo, lo interpretamos como velocidad
    if (incomingMessage[0] == 'a' && messageComplete && incomingMessage.indexOf('b') >= 0) {
      pos_b = incomingMessage.indexOf('b');
      motorSpeedIzq = incomingMessage.substring(1, pos_b).toInt(); // Convertimos el resto del mensaje a entero
      motorSpeedDer = incomingMessage.substring(pos_b + 1).toInt();
      c_motor0 = 0;
      c_motor1 = 0;
      Serial.print(motorSpeedIzq);                          // Mostramos velocidad por monitor serial
      Serial.print(", ");
      Serial.print(motorSpeedDer);
      Serial.print("\n");
    }
  }
      newtime = micros();

    //-----------------------------------
    // Ejemplo variable alterando cada 5 segundos


    //-----------------------------------
    // Actualizando Informacion de los encoders
    newposition0 = encoder0Pos;
    newposition1 = encoder1Pos;

    //-----------------------------------
    // Calculando Velocidad del motor en unidades de RPM
    float rpm = 31250;
    vel0 = (float)(newposition0 - oldposition0) * rpm / (newtime - time_ant); //RPM
    vel1 = (float)(newposition1 - oldposition1) * rpm / (newtime - time_ant); //RPM
    oldposition0 = newposition0;
    oldposition1 = newposition1;

    error_actual_motor0 = motorSpeedIzq - vel0;
    error_actual_motor1 = -motorSpeedDer - vel1;

    //Valores constantes K por cada motor del controlador PID
    Kp_motor0 = 0.1;
    Ki_motor0 = 0.025;
    Kd_motor0 = 0.0001;

    Kp_motor1 = 0.1;
    Ki_motor1 = 0.25;
    Kd_motor1 = 0.0001;
    
    //K0 para motor 0 y 1
    k0_motor0 = Kp_motor0 + dt * Ki_motor0 + Kd_motor0 / dt;
    k0_motor1 = Kp_motor1 + dt * Ki_motor1 + Kd_motor1 / dt;
    //K1 para motor 0 y 1
    k1_motor0 = -Kp_motor0 - (2 * Kd_motor0) / dt;
    k1_motor1 = -Kp_motor1 - (2 * Kd_motor1) / dt;
    //K2 para motor 0 y 1
    k2_motor0 = Kd_motor0 / dt;
    k2_motor1 = Kd_motor1 / dt;


    // El controlador PID en voltaje en el tiempo k
    c_motor0 = c_t1_motor0 + k0_motor0 * error_actual_motor0 + k1_motor0 * error_t1_motor0 + k2_motor0 * error_t2_motor0;
    c_motor1 = c_t1_motor1 + k0_motor1 * error_actual_motor1 + k1_motor1 * error_t1_motor1 + k2_motor0 * error_t2_motor1;

    //Error en tiempo k - 2 para motor 0 y 1
    error_t2_motor0 = error_t1_motor0;
    error_t2_motor1 = error_t1_motor1;
    //Error tiempo k - 1 para motor 0 y 1
    error_t1_motor0 = error_actual_motor0;
    error_t1_motor1 = error_actual_motor1;

    //c en tiempo k - 1 para motor 0 y 1
    c_t1_motor0 = c_motor0;
    c_t1_motor1 = c_motor1;

    //Evitamos que se pase de los 12 V de la bateria
    float V = 7.2;
    if (c_motor0 > V)
      c_motor0 = V;
    if (c_motor1 > V)
      c_motor1 = V;
    if (c_motor0 < -V)
      c_motor0 = -V;
    if (c_motor1 < -V)
      c_motor1 = -V;

    // setSpeed no recibe voltaje por lo que pasamos a su formato con el voltaje de nuestra bateria
    motorout0 = (c_motor0 / V) * 400;
    motorout1 = (c_motor1 / V) * 400;

    // Motor Voltage
    md.setM1Speed(motorout0); //MotorIzq
    md.setM2Speed(motorout1); //MotorDer

    time_ant = newtime;
    // Serial.print("MotorSpeed: ");
    // Serial.print(motorSpeedIzq);
    // Serial.print(", ");
    // Serial.print(motorSpeedDer);
    // Serial.print(" Vel0: ");
    Serial.print(vel0);
    Serial.print(",");
    // Serial.print(" Vel1: ");
    Serial.println(vel1);
    // Serial.print("\n");
    delay(1);
}

String readBluetoothMessage() {
  String buffer = "";  // Variable para construir el mensaje

  while (Serial3.available() > 0) {
    char receivedChar = Serial3.read();  // Leemos un carácter del puerto Bluetooth

    if (receivedChar == endChar) {
      messageComplete = true;            // Marcamos que el mensaje está completo
      break;                             // Salimos del ciclo de lectura
    } else {
      buffer += receivedChar;            // Agregamos el carácter al mensaje
    }

    delay(10);  // Pequeña espera para evitar errores de lectura
  }

  return buffer;  // Retornamos el mensaje leído
}
