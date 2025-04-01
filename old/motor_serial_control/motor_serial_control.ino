const int IN1 = 5;
const int IN2 = 4;
const int ENA = 6;
const int IN3 = 8;
const int IN4 = 7;
const int ENB = 9;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  Serial.begin(9600); // Set baud rate for communication
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); // Read the command character
    int speed = Serial.parseInt(); // Read the speed value (if any)

    // Process the received command
    switch (command) {
      case 'F': // Forward
        Motor1_Forward(speed);
        Motor2_Forward(speed);
        break;
      case 'B': // Backward
        Motor1_Backward(speed);
        Motor2_Backward(speed);
        break;
      case 'L': // Left
        Motor1_Brake();
        Motor2_Forward(speed);
        break;
      case 'R': // Right
        Motor1_Forward(speed);
        Motor2_Brake();
        break;
      case 'S': // Stop
        Motor1_Brake();
        Motor2_Brake();
        break;
      default:
        // Invalid command, stop the motors for safety
        Motor1_Brake();
        Motor2_Brake();
        break;
    }
  }
}

void Motor1_Forward(int Speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, Speed);
}

void Motor1_Backward(int Speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, Speed);
}

void Motor1_Brake() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void Motor2_Forward(int Speed) {
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, Speed);
}

void Motor2_Backward(int Speed) {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, Speed);
}

void Motor2_Brake() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
