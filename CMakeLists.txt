cmake_minimum_required(VERSION 3.21)
project(motion_detector1)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(motion_detector1 main.cpp)
target_link_libraries(motion_detector1 ${OpenCV_LIBS} )
