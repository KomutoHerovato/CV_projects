#include <iostream>
#include <stdexcept>

void processNumber(int num) {
    try {
        if (num == 0) {
            throw "Number is zero";
        } else if (num < 0) {
            throw std::out_of_range("Number is negative");
        } else if (num % 2 == 0) {
            throw std::logic_error("Number is even");
        }
        std::cout << "Number is valid" << std::endl;
    } catch (const char* msg) {
        std::cerr << "Caught exception: " << msg << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "Caught out_of_range exception: " << e.what() << std::endl;
    } catch (const std::logic_error& e) {
        std::cerr << "Caught logic_error exception: " << e.what() << std::endl;
    }
}

class CustomException : public std::runtime_error {
public:
    CustomException(const std::string& msg) : std::runtime_error(msg) {}
};

void processInput(int input) {
    if (input < 0) {
        throw CustomException("Negative input is not allowed");
    }
}

int main() {
    try {
        processNumber(0); // Генерируется исключение типа const char*
        processNumber(-5); // Генерируется исключение типа std::out_of_range
        processNumber(4); // Генерируется исключение типа std::logic_error
        processNumber(3); // Нет исключения
    } catch (...) {
        std::cerr << "Caught unknown exception" << std::endl;
    }

    try {
        processInput(-5); // Генерируется пользовательское исключение CustomException
    } catch (const CustomException& e) {
        std::cerr << "Caught custom exception: " << e.what() << std::endl;
    }

    return 0;
};



