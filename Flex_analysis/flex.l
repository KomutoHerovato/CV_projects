%option noyywrap

%{
#include <iostream>
#include "funct.h"
//string settings
#define MAX_STRING_LENGTH 255

//errors in string
#define ERROR_NULL_IN_STRING 0x11
#define ERROR_EOF_IN_STRING 0X12

//errors in comment
#define ERROR_EOF_IN_COMMENT 0X21
#define ERROR_NULL_IN_COMMENT 0X22

int rec_comment_num;

char  string_buffer[MAX_STRING_LENGTH];
char* string_pointer;

static void printLexeme(const char * tokName)
{
    printf("%s\n", tokName);
}
%}

DIGIT         [0-9]

TYPE_ID       [A-Z][A-Za-z0-9_]*
VAR           [a-z][A-Za-z0-9_]*

ELSE          [Ee][Ll][Ss][Ee]
IF            [Ii][Ff]
LOOP          [Ll][Oo][Oo][Pp]
THEN          [Tt][Hh][Ee][Nn]
WHILE         [Ww][Hh][Ii][Ll][Ee]
NOT           [Nn][Oo][Tt]
PRINT         [Pp][Rr][Ii][Nn][Tt]
PRINTLN       [Pp][Rr][Ii][Nn][Tt][Ll][Nn]

TRUE          [t][Rr][Uu][Ee]
FALSE         [f][Aa][Ll][Ss][Ee]

%x IN_COMMENT FOO_COMMENT STRING

%%
\"                                                  {
    string_pointer = string_buffer;
    BEGIN(STRING);
}

<STRING>{
[^<<EOF>>\0\"]+ {
    char* yptr = yytext;
    while (*yptr) {
        *string_pointer = *yptr;
        string_pointer++;
        yptr++;
    }
}

<<EOF>> {
    print_msg ("error::eof in string");
    return ERROR_EOF_IN_STRING;
}

\0 {
    print_msg ("error::null in string");
    return ERROR_NULL_IN_STRING;
}

\" {
    *string_pointer = '\0';
    print_msg ("Token class: STRING, Token value: ", string_buffer);
    BEGIN(INITIAL);
}
}

{DIGIT}+                                            {
    print_msg ("Token class: DIGIT, Token value: ", yytext);
}

{DIGIT}+"."{DIGIT}*                                 {
    print_msg ("Token class: FLOAT, Token value: ", yytext);
}

{ELSE}|{FALSE}|{IF}|{LOOP}|{THEN}|{WHILE}|{NOT}|{TRUE}|{PRINT}|{PRINTLN} {
    print_msg ("Token class: KEYWORD, Token value: ", yytext);
}

"("|")"|"{"|"}"|"+"|"-"|"/"|"*"|"="                 {
    print_msg ("Token class: SPEC_SYM, Token value: ", yytext);
}


{TYPE_ID}                                           {
    print_msg ("Token class: TYPE_ID, Token value: ", yytext);
}

{VAR}                                               {
    print_msg ("Token class: VAR, Token value: ", yytext);
}

"/*"                                                {
          rec_comment_num = 1;
          BEGIN(IN_COMMENT);
}

<IN_COMMENT>{
"*/"      {
            rec_comment_num--;

            if (rec_comment_num == 0) {
                BEGIN(INITIAL);
            }
}

[^*/<<EOF>>\0]+

"*"+[^/]

"/"+[^*]

"/*"    rec_comment_num++;

<<EOF>> {
    print_msg ("error::eof in comment");
    return ERROR_EOF_IN_COMMENT;
}

\0 {
    print_msg ("error::null in comment");
    return ERROR_NULL_IN_COMMENT;
}
}

"--"                                                {
    BEGIN(FOO_COMMENT);
}

<FOO_COMMENT>{
[^<<EOF>>\n]+  // eat comment

\n    BEGIN(INITIAL);

<<EOF>>  BEGIN(INITIAL);

}

[ \n\t]+

%%