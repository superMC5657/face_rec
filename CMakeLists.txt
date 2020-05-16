cmake_minimum_required(VERSION 3.5)
project(face_with_caffe)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR  /home/supermc/envs/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/supermc/envs/caffe-windows/include
        /usr/local/cuda/include
        include)


link_libraries("/home/supermc/envs/caffe-windows/build/lib/libcaffe.so")
link_libraries("/usr/lib/x86_64-linux-gnu/libglog.so"
        "/usr/lib/x86_64-linux-gnu/libboost_filesystem.so"
        "/usr/lib/x86_64-linux-gnu/libboost_system.so"
        "/usr/lib/x86_64-linux-gnu/libprotobuf.so.9")
#message
message(${CMAKE_CXX_COMPILER})
message(${CMAKE_C_COMPILER})
message(${OpenCV_DIR})
message(${OpenCV_INCLUDE_DIRS})


add_executable(face_with_caffe src/camera.cpp)
add_executable(make_csv src/csv.cpp)

target_link_libraries(face_with_caffe ${OpenCV_LIBS})
target_link_libraries(make_csv ${OpenCV_LIBS})