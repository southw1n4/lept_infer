#include <iostream>
#include <fstream>
#include <unistd.h>

#include "onnxparser/parser.h"


inline
void inverse(unsigned int* number) {
    char* d = (char*) number;
    std::swap(d[0], d[3]);
    std::swap(d[1], d[2]);
}

class Mnist{
public:
    Mnist(const std::string& path) {
        std::string label_path_ = path + "/t10k-labels.idx1-ubyte";
        std::string image_path_ = path + "/t10k-images.idx3-ubyte";

        if(access(image_path_.c_str(), F_OK) != 0 || access(label_path_.c_str(), F_OK) != 0) {
            std::cout << "file not exists" << std::endl;
        }

        image_.open(image_path_);
        label_.open(label_path_);

        image_.read((char *)&magic_number_, 4);
        image_.read((char *)&len_, 4);

        inverse(&magic_number_);
        inverse(&len_);


    }

    ~Mnist() {
        image_.close();
        label_.close();
    }

    std::pair<leptinfer::Tensor, int> operator()(int idx) {
        if(idx > len_) {
            std::cout << "error" << std::endl;
        }

        leptinfer::Tensor img({1, 1, 28, 28});
        image_.seekg(16 + idx * 28 * 28, std::ios::beg);
        label_.seekg(8 + idx, std::ios::beg);
        char* data = (char *)img.data();
        float* ptr = (float*)img.data();
        uint8_t lab;

        image_.read((char *)data, 28 * 28);
        label_.read((char *)&lab, 1);

        for(int i = 28 * 28 - 1; i >= 0; -- i) {
            ptr[i] = (float)(uint8_t)data[i];
        }
        return std::make_pair(img, (int)lab);
    }

private:
    std::ifstream image_;
    std::ifstream label_;
    unsigned int len_;
    unsigned int magic_number_;
};

int get_max(leptinfer::Tensor& a) {
    std::cout << a << std::endl;

    int p = 0;
    for(int i = 0;i < 10; ++ i) {
        if(a({1, i}) > a({1, p})) p = i;
    }
    return p;
}
using leptinfer::sub;
using leptinfer::div;
using leptinfer::Tensor;

int main(int argc, char* argv[]) {
std::string model_path = "../models/mnist.onnx";
    if(argc > 1) {
        model_path = argv[1];
    }
    auto parser = new leptinfer::OnnxParser();
    auto net = parser->parse(model_path);
    Mnist test_data("../data");  

    Tensor mean = Tensor({1, 1, 1, 1}, leptinfer::Tensor::ETENOSR_TYPE::TYPE_FP32, 0.1307);
    Tensor var = Tensor({1, 1, 1, 1}, leptinfer::Tensor::ETENOSR_TYPE::TYPE_FP32, 0.3081);
    int test_num = 10;
    for(int i = 0; i < test_num; ++ i) {

        auto data_i = test_data(i);
        std::cout << data_i.first << std::endl;
        auto predict = net->inference(div(sub(data_i.first, mean), var));

        int p = get_max(predict[0]);

        std::cout << "label:   " << data_i.second << std::endl;
        std::cout << "predict: " << p << std::endl;
    }


    delete parser;
    delete net;
    return 0;
}
