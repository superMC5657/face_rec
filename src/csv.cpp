/* -*- coding: utf-8 -*-
  !@time: 19-7-29 上午1:56
  !@author: superMC @email: 18758266469@163.com
  !@fileName: csv.cpp
 */

#include "utils.cpp"

class Csv {
private:
    FaceNet faceNet;
    MTCNN mtcnn;

    // 读取image_dir下的全部文件,并排序.
    void get_files(const string &imageDir_path, vector<string> &files) {
        DIR *dir;
        struct dirent *ptr;
        char base[1000];
        if ((dir = opendir(imageDir_path.c_str())) == NULL) {
            perror("Open dir error...");
            exit(1);
        }
        while ((ptr = readdir(dir)) != NULL) {
            if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
                continue;
            else if (ptr->d_type == 8)    ///file
                //printf("d_name:%s/%s\n",basePath,ptr->d_name);
                files.emplace_back(ptr->d_name);
            else if (ptr->d_type == 10)    ///link file
                //printf("d_name:%s/%s\n",basePath,ptr->d_name);
                continue;
            else if (ptr->d_type == 4)    ///dir
            {
                files.emplace_back(ptr->d_name);
            }
        }
        closedir(dir);
        sort(files.begin(), files.end());
    }

public:
    //初始化faceNet 和mtcnn
    void init_model(const string &model_path) {
        faceNet.init_model(model_path);
        mtcnn.init_model("../data");
    }

    // 传入文件夹路径,经过mtcnn,faceNet得到features写入csv.
    void to_csv(const string &csv_path, const string &imageDir_path) {
        vector<string> files;
        vector<string> labels;
        get_files(imageDir_path, files);
        vector<vector<float>> features;
        for (const string& file:files) {
            vector<vector<float>> features_copy;
            Mat image = imread(imageDir_path + '/' + file);
            vector<array<int, 4>> coordinates;
            int single = get_features(faceNet, mtcnn, image, features_copy, coordinates);
            if (single == 1) {
                labels.push_back(file.substr(0, file.length() - 4));
                features.push_back(features_copy[0]);
            } else {
                cout << file + "文件异常" << endl;
            }
        }
        ofstream outFile;
        outFile.open(csv_path, ios::out);
        for (int i = 0; i < features.size(); i++) {
            vector<float> feature = features[i];
            string label = labels[i];
            outFile << label << ',';
            for (int j = 0; j < FEATURES_NUM; j++) {
                outFile << feature[j] << ',';
            }
            outFile << endl;
        }
        outFile.close();
    }

};


int main(int argc, char *argv[]) {
    GlobalInit(&argc, &argv);
    Csv csv;
    string model_path = argv[1];
    string imageDir_path = argv[2];
    string csv_path = argv[3];
    csv.init_model(model_path);
    csv.to_csv(csv_path, imageDir_path);
}
