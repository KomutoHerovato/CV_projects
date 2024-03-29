all: flex run

flex: flex.l funct.cpp
	flex --header-file=lex_lib.hpp flex.l funct.cpp

run: main.cpp lex.yy.c
	g++ main.cpp lex.yy.c

clean:
	rm *.out lex.yy.c lex_lib.hpp