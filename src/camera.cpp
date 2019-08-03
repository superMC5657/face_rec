
#include "utils.cpp"

class FaceRec {
private:
    FaceNet faceNet;
    MTCNN mtcnn;
    vector<float *> csv_features;
    vector<string> csv_labels;


public:

    void init_model(const string &model_path) {
        faceNet.init_model(model_path);
        mtcnn.init_model("../data");

    }

    void init_face_database(const string &face_database_path) {
        to_features(face_database_path, csv_features, csv_labels);
    }

    void face_recog_img(const string &img_path) {
        Mat image = imread(img_path);
        vector<float *> features;
        vector<array<int, 4>> coordinates;
        get_features(faceNet, mtcnn, image, features, coordinates, false);
        int num = features.size();
        vector<string> labels(num);
        for (int i = 0; i < num; i++) {
            to_label(csv_features, csv_labels, features[i], labels[i]);
            //            cout << cosine << endl;
            cout << labels[i] << "\t";
            //画框
            rectangle(image, Rect(coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3]),
                      Scalar(255, 0, 0), 2);
            //label
            float scale = coordinates[i][2] * 0.005f;
            putText(image, labels[i], Point2d(coordinates[i][0], coordinates[i][1]), FONT_HERSHEY_SIMPLEX, scale,
                    Scalar(0, 255, 0), 2);
        }
        imwrite("../final.png", image);

    }

    void face_recog_video(Mat &im_in, int &face_id) {
        vector<float *> features;
        vector<array<int, 4>> coordinates;
        get_features(faceNet, mtcnn, im_in, features, coordinates, false);
    }


};


int main(int argc, char *argv[]) {
    //close caffe's log
    GlobalInit(&argc, &argv);
    if (argc < 3) {
        printf("input error, must have 3 params");
    }
    //"../data/MobileFaceNet_112x112_02_gray_500.caffemodel"
    string model_path = argv[1];
    string csv_path = argv[2];
    string image_path = argv[3];
    FaceRec faceRec;
    faceRec.init_face_database(csv_path);
    faceRec.init_model(model_path);
    faceRec.face_recog_img(image_path);
    return 0;
}