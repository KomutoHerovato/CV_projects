cmake_minimum_required(VERSION 3.26)
project(List)

set(CMAKE_CXX_STANDARD 17)

add_executable(List main.cpp
        List.cpp
        ListItem.cpp)
