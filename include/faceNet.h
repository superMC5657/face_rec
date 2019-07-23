//
// Created by zhangyan on 19-6-29.
//

#ifndef FACE_WITH_CAFFE_FACENET_H
#define FACE_WITH_CAFFE_FACENET_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

const int CHANNEL = 3;
const int WIDTH = 112;
const int HEIGHT = 112;
//mean & std
const float mean_val = 127.5f;
const float std_val = 0.0078125f;
using namespace caffe;
using namespace std;
using namespace cv;



#endif //FACE_WITH_CAFFE_FACENET_H
