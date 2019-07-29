#include "utils.cpp"



class FaceRec {
private:
    FaceNet faceNet;
    MTCNN mtcnn;

public:

    void init_model(const string &model_path) {
        faceNet.init_model(model_path);
        mtcnn.init_model("../data");
    }

    void init_face_database(const string &face_database_path) {

    }

    void face_recog_img(const string &img_path) {
        Mat image = imread(img_path);
        vector<float *> features;
        get_features(faceNet, mtcnn, image, features, false);
        for (int i = 0; i < features.size() - 1; i++) {
            float cosine = get_cosine(features[i], features[features.size() - 1]);
            printf("%.3lf\n", cosine);
            //            cout << cosine << endl;
        }
    }

    void face_recog_video(Mat &im_in, int &face_id) {
        vector<float *> features;
        get_features(faceNet, mtcnn, im_in, features, false);
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
    string image_path = argv[2];
    //"../data/one_man_img.csv"
    FaceRec faceRec;
    faceRec.init_model(model_path);
    faceRec.face_recog_img(image_path);
    return 0;
}