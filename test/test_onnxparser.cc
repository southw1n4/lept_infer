

#include "onnxparser/parser.h"

using namespace leptinfer;

void test_onnxparser(){
    const std::string model_path = "../models/mnist.onnx";
    auto onnx_parser = new OnnxParser();
    auto net = onnx_parser->parse(model_path);

    delete net;
    delete onnx_parser;

}
