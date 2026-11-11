#include "protocol.hpp"

Controller controller;
char frame[160];
char response[200];
size_t used = 0;
bool overflowed = false;

void setup() {
    Serial.begin(115200);
}

void loop() {
    while (Serial.available()) {
        const char value = Serial.read();
        if (value == '\n') {
            if (overflowed) {
                Serial.println("ERROR,frame too long");
            } else {
                frame[used] = '\0';
                controller.respond(frame, response, sizeof(response));
                Serial.println(response);
            }
            used = 0;
            overflowed = false;
        } else if (used < sizeof(frame) - 1) {
            frame[used++] = value;
        } else {
            overflowed = true;
        }
    }
}
