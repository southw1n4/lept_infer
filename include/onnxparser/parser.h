#ifndef __ONNXPARSER_PARSER_H__
#define __ONNXPARSER_PARSER_H__

#include <unordered_map>

#include "net/netbase.h"
#include "operator/opbase.h"
#include "onnxparser/onnx.proto3.pb.h"



namespace leptinfer{
class OnnxParser{
public:
    OnnxParser(){};
    ~OnnxParser(){};
    Net* parse(const std::string&, bool varbose = true);

private:
    bool parser_op(::onnx::GraphProto&);
    bool parser_tensor(::onnx::GraphProto&);
private:

    Net* net_ = NULL;
    bool varbose_;
    std::unordered_map<std::string, Tensor*>  named_tensors_;
    std::unordered_map<std::string, Op*> named_ops_;
    std::unordered_map<std::string, bool> output_name_;


};
}


#endif
