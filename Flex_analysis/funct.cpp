//
// Created by Дамир Мухаметзянов on 3/29/24.
//

#include "funct.h"

void print_msg (const char* my_msg, char* yytext) {
    if (yytext != nullptr) {
        std::cout << my_msg << yytext << std::endl;
    } else {
        std::cout << my_msg << std::endl;
    }
}