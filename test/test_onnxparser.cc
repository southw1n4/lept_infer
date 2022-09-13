

#include "onnxparser/parser.h"

using namespace leptinfer;

void test_onnxparser(){
    const std::string model_path = "/home/southw1nd/download/model_13.onnx";
    auto onnx_parser = new OnnxParser();
    auto net = onnx_parser->parse(model_path, true);

    auto x = Tensor({1, 3, 256, 256});
    auto y = net->inference(x);

    std::cout << y[0] << std::endl;

    delete net;
    delete onnx_parser;


}
