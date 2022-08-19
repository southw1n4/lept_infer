

#include "onnxparser/parser.h"

using namespace leptinfer;

void test_onnxparser(){
    const std::string model_path = "../models/mnist.onnx";
    auto onnx_parser = new OnnxParser();
    auto net = onnx_parser->parse(model_path);
    auto output = net->excute(Tensor({1, 1, 28, 28}));


    std::cout << output[0] << std::endl;
    delete net;
    delete onnx_parser;


}
