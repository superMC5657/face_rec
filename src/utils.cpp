/* -*- coding: utf-8 -*-
  !@time: 19-7-26 上午1:26
  !@author: superMC @email: 18758266469@163.com
  !@fileName: utils.cpp
 */
#include "mtcnn.cpp"
#include "faceNet.cpp"


float FACTOR = 0.709f;
float THRESHOLD[3] = {0.6f, 0.6f, 0.6f};
int MINSIZE = 40;

float get_cosine(const float *new_features, const float *old_features) {
    float new_mold = 0;
    float old_mold = 0;
    float innerProduct = 0;
    for (int i = 0; i < FEATURES_NUM; i++) {
        new_mold += new_features[i] * new_features[i];
        old_mold += old_features[i] * old_features[i];
        innerProduct += new_features[i] * old_features[i];
    }
    float cosine = 1 - innerProduct / sqrt(new_mold * old_mold);
    return cosine;
}

array<int, 4> add_margin(FaceBox bbox, float margin) {
    array<int, 4> a{};
    a[2] = (int) ((bbox.xmax - bbox.xmin + 1) * (1 + margin));
    a[3] = (int) ((bbox.ymax - bbox.ymin + 1) * (1 + margin));
    a[0] = (int) (bbox.xmin - margin * a[2] * 0.5);
    a[1] = (int) (bbox.ymin - margin * a[3] * 0.5);
    return a;
}

Mat getwarpAffineImg(Mat &src, int le_landmark_x, int le_landmark_y, int re_landmark_x, int re_landmark_y) {

    //计算两眼中心点,按照此中心点进行旋转
    Point2f eyesCenter = Point2f((le_landmark_x + re_landmark_x) * 0.5f, (le_landmark_y + re_landmark_y) * 0.5f);

    // 计算两个眼睛间的角度
    double dy = (re_landmark_y - le_landmark_y);
    double dx = (re_landmark_x - le_landmark_x);
    double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

    //由eyesCenter, andle, scale按照公式计算仿射变换矩阵，此时1.0表示不进行缩放
    Mat rot = getRotationMatrix2D(eyesCenter, angle, 1.0);
    // 进行仿射变换，变换后大小为src的大小
    warpAffine(src, rot, rot, src.size());
    return rot;
}

int getImage(MTCNN &mtcnn, const Mat &image, vector<Mat> &faces,bool make_csv, float margin ) {
    vector<FaceInfo> faceInfos = mtcnn.Detect(image, MINSIZE, THRESHOLD, FACTOR, 3);
    int num = faceInfos.size();
    if (num == 0) {
        return 0;
    }
    if (make_csv) {
        if (num == 1) {
            for (FaceInfo &faceInfo : faceInfos) {
                array<int, 4> a{};
                a = add_margin(faceInfo.bbox, margin);
                Mat croppedImage(image, Rect(a[0], a[1], a[2], a[3]));

                int le_landmark_x = (int) faceInfo.landmark[0] - a[0];
                int le_landmark_y = (int) faceInfo.landmark[1] - a[1];
                int re_landmark_x = (int) faceInfo.landmark[2] - a[0];
                int re_landmark_y = (int) faceInfo.landmark[3] - a[1];
                croppedImage = getwarpAffineImg(croppedImage, le_landmark_x, le_landmark_y, re_landmark_x,
                                                re_landmark_y);
                cv::resize(croppedImage, croppedImage, Size(112, 112));
                Mat single_gray_image;
                cv::cvtColor(croppedImage, single_gray_image, COLOR_BGR2GRAY);
                vector<Mat> channels(CHANNEL);
                for (int i = 0; i < CHANNEL; i++) {
                    channels[i] = single_gray_image;
                }
                merge(channels, croppedImage);
                faces.push_back(croppedImage);
            }
        }
    }
    else {
        for (FaceInfo &faceInfo : faceInfos) {
            array<int, 4> a{};
            a = add_margin(faceInfo.bbox, margin);
            Mat croppedImage(image, Rect(a[0], a[1], a[2], a[3]));

            int le_landmark_x = (int) faceInfo.landmark[0] - a[0];
            int le_landmark_y = (int) faceInfo.landmark[1] - a[1];
            int re_landmark_x = (int) faceInfo.landmark[2] - a[0];
            int re_landmark_y = (int) faceInfo.landmark[3] - a[1];
            croppedImage = getwarpAffineImg(croppedImage, le_landmark_x, le_landmark_y, re_landmark_x,
                                            re_landmark_y);
            cv::resize(croppedImage, croppedImage, Size(112, 112));
            Mat single_gray_image;
            cv::cvtColor(croppedImage, single_gray_image, COLOR_BGR2GRAY);
            vector<Mat> channels(CHANNEL);
            for (int i = 0; i < CHANNEL; i++) {
                channels[i] = single_gray_image;
            }
            merge(channels, croppedImage);
            faces.push_back(croppedImage);
        }
    }
    return num;
}

int get_features(FaceNet &faceNet, MTCNN &mtcnn, Mat &image, vector<float *> &features,bool make_csv= true, float margin = 0.2) {
    vector<Mat> faces;
    int single = 0;
    single = getImage(mtcnn, image, faces,make_csv, margin);
    if (make_csv){
        if (single==1) {
            faceNet.to_features(faces, features);
        }

    } else{
        if (single>0){
            faceNet.to_features(faces,features);
        }
    }
    return single;
}

