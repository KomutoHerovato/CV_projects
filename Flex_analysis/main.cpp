//
// Created by Дамир Мухаметзянов on 3/29/24.
//

#include "lex_lib.hpp"

int main (int argc, char** argv) {
    if (argc > 1) {
        yyin = fopen (argv[1], "r");
    } else {
        yyin = stdin;
    }

    yylex();
}

