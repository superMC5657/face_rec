/* -*- coding: utf-8 -*-
  !@time: 19-7-29 上午1:42
  !@author: superMC @email: 18758266469@163.com
  !@fileName: envs.h
 */
#ifndef FACE_WITH_CAFFE_ENVS_H
#define FACE_WITH_CAFFE_ENVS_H

#include <vector>
#include <caffe/caffe.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <boost/shared_ptr.hpp>
#include <unistd.h>
#include <dirent.h>

using namespace caffe;
using namespace std;
using namespace cv;

const float mean_val = 127.5f;
const float std_val = 0.0078125f;

#endif //FACE_WITH_CAFFE_ENVS_H