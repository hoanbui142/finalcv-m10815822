cmake_minimum_required(VERSION 3.10)

project(cv-final-m10815822)

set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(finalcv-m10815822
    src/finalcv-m10815822.cpp
    src/reconstruct3d.cpp
)

target_link_libraries(finalcv-m10815822 ${OpenCV_LIBS})