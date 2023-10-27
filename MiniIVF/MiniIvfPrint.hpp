#ifndef INCLUDE_MINIIVFPRINT_HPP
#define INCLUDE_MINIIVFPRINT_HPP

#include <iostream>
#include <string>

namespace MiniIVF {

enum COLOR {
    RED,
    GREEN,
    YELLOW,
    BLUE,
    DEFAULT
};

class Printer {
public: /// variables

public: /// methods
    static void print(const std::string & str, COLOR color = COLOR::DEFAULT) {
        switch (color) {
            case COLOR::RED:
                std::cout << ("\033[31;1m" + str + "\033[0m");
                break;
            case COLOR::GREEN:
                std::cout << ("\033[32;1m" + str + "\033[0m");
                break;
            case COLOR::YELLOW:
                std::cout << ("\033[33;1m" + str + "\033[0m");
                break;
            case COLOR::BLUE:
                std::cout << ("\033[34;1m" + str + "\033[0m");
                break;
            default:
                std::cout << str;
        }
    }

    static void warning(const std::string & str) {
        std::cout << ("\033[33;1m" + str + "\033[0m");
    }

    static void err(const std::string & str) {
        std::cout << ("\033[31;1m" + str + "\033[0m");
    }

};

};

#endif // ! INCLUDE_MINIIVFPRINT_HPP