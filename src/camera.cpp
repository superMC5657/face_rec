
#include <iostream>
#include "caffe/caffe.hpp"
#include <vector>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <cublas_v2.h>
#include <format.hpp>
#include "faceNet.cpp"
#include "mtcnn.cpp"
#include <glog/logging.h>


float FACTOR = 0.709f;
float THRESHOLD[3] = {0.6f, 0.6f, 0.6f};
int MINSIZE = 40;


class FaceRec {
private:
    FaceNet faceNet;
    MTCNN mtcnn;
    vector<float *> csv_features;

    array<int, 4> add_margin(FaceBox bbox, float margin) {
        array<int, 4> a{};
        a[2] = (int) ((bbox.xmax - bbox.xmin + 1) * (1 + margin));
        a[3] = (int) ((bbox.ymax - bbox.ymin + 1) * (1 + margin));
        a[0] = (int) (bbox.xmin - margin * a[2] * 0.5);
        a[1] = (int) (bbox.ymin - margin * a[3] * 0.5);
        return a;
    }

    Mat getwarpAffineImg(Mat &src, int le_landmark_x, int le_landmark_y, int re_landmark_x, int re_landmark_y) {

        //计算两眼中心点，按照此中心点进行旋转， 第31个为左眼坐标，36为右眼坐标
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

    void get_features(const Mat &image, vector<float *> &features, float margin = 0.2) {
        vector<Mat> faces;
        getImage(image, faces, margin);
        faceNet.to_features(faces, features);
    }

    void getImage(const Mat &image, vector<Mat> &faces, float margin) {
        vector<FaceInfo> faceInfos = mtcnn.Detect(image, MINSIZE, THRESHOLD, FACTOR, 3);
        for (FaceInfo &faceInfo : faceInfos) {
            array<int, 4> a{};
            a = add_margin(faceInfo.bbox, margin);
            Mat croppedImage(image, Rect(a[0], a[1], a[2], a[3]));

            int le_landmark_x = (int) faceInfo.landmark[0] - a[0];
            int le_landmark_y = (int) faceInfo.landmark[1] - a[1];
            int re_landmark_x = (int) faceInfo.landmark[2] - a[0];
            int re_landmark_y = (int) faceInfo.landmark[3] - a[1];
            croppedImage = getwarpAffineImg(croppedImage, le_landmark_x, le_landmark_y, re_landmark_x, re_landmark_y);
            cv::resize(croppedImage, croppedImage, Size(112, 112));
            Mat single_gray_image;
            cv::cvtColor(croppedImage, single_gray_image, COLOR_BGR2GRAY);
            vector<Mat> channels(3);
            channels[0] = single_gray_image;
            channels[1] = single_gray_image;
            channels[2] = single_gray_image;
            merge(channels, croppedImage);
            faces.push_back(croppedImage);
        }
    }

public:

    int init_model(const string &model_path) {
        faceNet.init_model(model_path);
        mtcnn.init_model("../data");
        return 0;
    }

    int init_face_database(const string &face_database_path) {


        return 0;
    }

    int face_recog_img(const string &img_path) {
        Mat image = imread(img_path);
        vector<float *> features;
        get_features(image, features);
        for (float *feature : features) {
            for (int i = 0; i < 128; i++) {
                cout << *(feature + i);
            }
            cout << endl;
        }
    }

    int face_recog_video(const Mat &im_in, int &face_id) {
        vector<float *> features;
        get_features(im_in, features);
    }


};


int main(int argc, char *argv[]) {
    //close caffe's log
    GlobalInit(&argc, &argv);
    if (argc < 3) {
        printf("input error, must have two params");
    }
    //"../data/MobileFaceNet_112x112_02_gray_500.caffemodel"
    string model_path = argv[1];
    //"../data/one_man_img.csv"
    string image_path = argv[2];
    FaceRec faceRec;
    faceRec.init_model(model_path);
    faceRec.face_recog_img(image_path);


    return 0;
}