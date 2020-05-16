//
// Created by zhangyan on 19-6-29.
//

#ifndef FACE_WITH_CAFFE_FACENET_H
#define FACE_WITH_CAFFE_FACENET_H


#include <envs.h>

const int CHANNEL = 3;
const int WIDTH = 112;
const int HEIGHT = 112;
const int FEATURES_NUM = 128;

class FaceNet {
private:
    boost::shared_ptr<Net<float>> net;
    // 将通道放到第二维度

public:
    void init_model(const string &model_path);

    void to_features(vector<Mat> &imgs, vector<vector<float>> &features);

};

#endif //FACE_WITH_CAFFE_FACENET_H
