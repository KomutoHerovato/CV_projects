//
// Created by Дамир Мухаметзянов on 3/29/24.
//

#ifndef FLEX_ANALYSIS_FUNCT_H
#define FLEX_ANALYSIS_FUNCT_H


#include <iostream>
//string settings
#define MAX_STRING_LENGTH 255

//errors in string
#define ERROR_NULL_IN_STRING 0x11
#define ERROR_EOF_IN_STRING 0X12

//errors in comment
#define ERROR_EOF_IN_COMMENT 0X21
#define ERROR_NULL_IN_COMMENT 0X22



//template function for messages for all things
void print_msg (const char* my_msg, char* yytext = nullptr);



#endif //FLEX_ANALYSIS_FUNCT_H
