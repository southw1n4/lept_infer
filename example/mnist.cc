#include <iostream>
#include <fstream>

#include "onnxparser/parser.h"


inline
void inverse(int* number) {
    char* d = (char*) number;
    std::swap(d[0], d[3]);
    std::swap(d[1], d[2]);
}

class Mnist{
public:
    Mnist(const std::string& path) {
        std::string label_path_ = path + "/t10k-labels-idx1-ubyte.gz";
        std::string image_path_ = path + "/t10k-images-idx3-ubyte.gz";

        image_.open(image_path_);
        label_.open(label_path_);

        image_ >> magic_number_ >> len_;
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
        image_.seekg(8 + idx * 28 * 28, std::ios::beg);
        label_.seekg(8 + idx, std::ios::beg);
        char* data = (char *)img.data();
        float* ptr = (float*)data + 28 * 28;

        image_.read(data, 28 * 28);
        for(int i = 28 * 28 - 1; i >= 0; i --) {
            ptr[i] = ;

            /*TODO
             *
             */
        }

    }

private:
    std::ifstream image_;
    std::ifstream label_;
    int len_;
    int magic_number_;
};


int main(int argc, char* argv[]) {

    std::string model_path = "../models/mnist.onnx";
    if(argc > 1) {
        model_path = argv[1];
    }
    auto parser = new leptinfer::OnnxParser();
    auto net = parser->parse(model_path);
    auto mnist_test_data = Mnist("./data");  



    delete parser;
    delete net;
    return 0;
}
