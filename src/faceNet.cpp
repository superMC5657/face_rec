//
// Created by zhangyan on 19-6-29.
//

#include "faceNet.h"

// 将通道放到第二维度
void FaceNet::init_model(const string &model_path) {
    string proto = "../data/deploy.prototxt";
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);
    net.reset(new Net<float>(proto, TEST));
    net->CopyTrainedLayersFrom(model_path);
    Blob<float> *input_layer = net->input_blobs()[0];
    input_layer->Reshape(0, CHANNEL, HEIGHT, WIDTH);
}

void FaceNet::to_features(vector<Mat> &imgs, vector<vector<float>> &features) {
    Blob<float> *input_layer = net->input_blobs()[0];
    Blob<float> *output = nullptr;
    input_layer->Reshape(imgs.size(), CHANNEL, WIDTH, HEIGHT);
    int spatial_size = WIDTH * HEIGHT;
    for (int i = 0; i < imgs.size(); i++) {
        float *input_data = input_layer->mutable_cpu_data();
        float *input_data_n = input_data + input_layer->offset(i);
        Vec3b *roi_data = (Vec3b *) imgs[i].data;
        CHECK_EQ(imgs[i].isContinuous(), true);
        for (int k = 0; k < spatial_size; ++k) {
            input_data_n[k] = float((roi_data[k][0] - mean_val) * std_val);
            input_data_n[k + spatial_size] = float((roi_data[k][1] - mean_val) * std_val);
            input_data_n[k + 2 * spatial_size] = float((roi_data[k][2] - mean_val) * std_val);
        }
    }
    net->Forward();
    output = net->blob_by_name("fc5").get();
    const float *confidence_data = output->cpu_data();
    for (int i = 0; i < imgs.size(); i++) {
        vector<float> feature(FEATURES_NUM);
        for (int j = 0; j < FEATURES_NUM; j++) {
            feature[j] = confidence_data[i * FEATURES_NUM + j];
        }
        features.push_back(feature);
    }
}
