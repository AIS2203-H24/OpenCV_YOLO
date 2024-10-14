cmake_minimum_required(VERSION 3.19)
project(OpenCV_yolo)

set(CMAKE_CXX_STANDARD 20)

if (VCPKG_TARGET_TRIPLET)
    set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
else ()
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif ()

find_package(OpenCV CONFIG REQUIRED)

add_executable(OpenCV_yolo main.cpp)
target_link_libraries(OpenCV_yolo PRIVATE ${OpenCV_LIBRARIES})

file(COPY "data/" DESTINATION "${CMAKE_BINARY_DIR}/data")
