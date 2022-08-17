#ifndef __ONNXPARSER_PARSER_HELPER_H__
#define __ONNXPARSER_PARSER_HELPER_H__

#include <unordered_map>
#include <vector>

#include "operator/opbase.h"
#include "onnxparser/onnx.proto3.pb.h"

#define __SUPPORT_OP_NAME \
    "Conv"

#define __SUPPORT_OP_FUNC \
    parse_conv 


namespace leptinfer{

using OP_FUNC_PTR = Op*(*)(::onnx::NodeProto&,
                           std::unordered_map<std::string, Tensor*>&,
                           std::unordered_map<std::string, Op*>&); 
using OP_FUNC_NAME = std::string;




Op* parse_conv(::onnx::NodeProto& node,
                std::unordered_map<std::string, Tensor*>& named_tensors,
                std::unordered_map<std::string, Op*>& named_ops);


const std::vector<OP_FUNC_PTR> SUPPROT_OP_FUNC = {__SUPPORT_OP_FUNC};
const std::vector<OP_FUNC_NAME> SUPPORT_OP_NAME = {__SUPPORT_OP_NAME};
}


#endif
