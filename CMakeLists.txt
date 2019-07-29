cmake_minimum_required(VERSION 3.5)
project(face_with_caffe)
set(CMAKE_CXX_STANDARD 14)
include_directories(/home/zhangyan/caffe-windows/include
        /usr/local/cuda/include
        /usr/include/boost
        /usr/local/include
        include)
find_library(caffe /home/zhangyan/caffe-windows/build/lib)
find_library(opencv /usr/local/lib)

link_libraries("/home/zhangyan/caffe-windows/build/lib/libcaffe.so"
        "/usr/lib/x86_64-linux-gnu/libglog.so"
        "/usr/lib/x86_64-linux-gnu/libboost_filesystem.so"
        "/usr/lib/x86_64-linux-gnu/libboost_system.so"
        "/usr/lib/x86_64-linux-gnu/libprotobuf.so.9"
        "/usr/lib/x86_64-linux-gnu/libpng16.so.16")

link_libraries("/usr/local/lib/libopencv_calib3d.so"
        "/usr/local/lib/libopencv_core.so"
        "/usr/local/lib/libopencv_dnn.so"
        "/usr/local/lib/libopencv_features2d.so"
        "/usr/local/lib/libopencv_flann.so"
        "/usr/local/lib/libopencv_highgui.so"
        "/usr/local/lib/libopencv_imgproc.so"
        "/usr/local/lib/libopencv_ml.so"
        "/usr/local/lib/libopencv_imgcodecs.so"
        "/usr/local/lib/libopencv_objdetect.so"
        "/usr/local/lib/libopencv_photo.so"
        "/usr/local/lib/libopencv_stitching.so"
        "/usr/local/lib/libopencv_superres.so"
        "/usr/local/lib/libopencv_videoio.so"
        "/usr/local/lib/libopencv_video.so"
        "/usr/local/lib/libopencv_videostab.so")

link_directories("/usr/local/lib")

add_executable(face_with_caffe src/camera.cpp)
add_executable(make_csv src/csv.cpp)