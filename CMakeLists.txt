cmake_minimum_required(VERSION 3.12)
project(Global-Image-Registration-for-HDR-Images-Using-Ward-s-MTB)

set(CMAKE_CXX_STANDARD 14)

add_executable(run cpu/main.cpp cpu/Image.cpp)