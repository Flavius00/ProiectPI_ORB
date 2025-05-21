cmake_minimum_required(VERSION 3.14)
project(ProiectPI)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Creează executabilul
add_executable(traffic_sign_detection
        main.cpp
        shape_detector.cpp
        sign_detector.cpp
        opencv_detector.cpp
        shape_type.h
)

# Leagă bibliotecile OpenCV
target_link_libraries(traffic_sign_detection ${OpenCV_LIBS})

set_target_properties(traffic_sign_detection PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)