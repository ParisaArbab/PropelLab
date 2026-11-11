#include <iostream>
#include <string>

#include "bench/protocol.hpp"

int main() {
    Controller controller;
    std::string line;
    char response[200];
    while (std::getline(std::cin, line)) {
        controller.respond(line.c_str(), response, sizeof(response));
        std::cout << response << std::endl;
    }
    return 0;
}
