cmake_minimum_required(VERSION 3.12)

project(Game2048)

set(CMAKE_CXX_STANDARD 14)

find_library(SFML_GRAPHICS_LIBRARY NAMES sfml-graphics HINTS "/usr/lib/x86_64-linux-gnu")
find_library(SFML_WINDOW_LIBRARY NAMES sfml-window HINTS "/usr/lib/x86_64-linux-gnu")
find_library(SFML_SYSTEM_LIBRARY NAMES sfml-system HINTS "/usr/lib/x86_64-linux-gnu")

find_path(SFML_INCLUDE_DIR NAMES SFML/Graphics.hpp HINTS "/usr/include")

include_directories(${SFML_INCLUDE_DIR})

add_executable(Game2048 2048.cpp)

target_link_libraries(Game2048 ${SFML_GRAPHICS_LIBRARY} ${SFML_WINDOW_LIBRARY} ${SFML_SYSTEM_LIBRARY})