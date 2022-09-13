#ifndef __ONNXPARSER_PARSER_HELPER_H__
#define __ONNXPARSER_PARSER_HELPER_H__

#include <unordered_map>
#include <vector>

#include "operator/opbase.h"
#include "onnxparser/onnx.proto3.pb.h"



#define __SUPPORT_OP_NAME \
    "Conv", "Relu", "Gemm", "Flatten", "Add", "Clip", "Div", "Constant", "Mul", "GlobalAveragePool", "BatchNormalization", "Elu", "Concat"

#define __SUPPORT_OP_FUNC \
    parse_conv, parse_relu, parse_gemm,     parse_flatten, parse_add,\
    parse_clip, parse_div,  parse_constant, parse_mul,     parse_avg_pool,\
    parse_batch_norm, parse_elu, parse_cat

#define REGISTER(name) \
        Op* parse_##name(::onnx::NodeProto&, \
                     std::unordered_map<std::string, Tensor*>&);


namespace leptinfer{

REGISTER(relu)
REGISTER(conv)
REGISTER(gemm)
REGISTER(flatten)
REGISTER(add)
REGISTER(clip)
REGISTER(div)
REGISTER(constant)
REGISTER(mul)
REGISTER(avg_pool)
REGISTER(batch_norm)
REGISTER(elu)
REGISTER(cat)


static std::vector<Op*(*)(::onnx::NodeProto&, \
                          std::unordered_map<std::string, Tensor*>&)> \
                                SUPPROT_OP_FUNC = {__SUPPORT_OP_FUNC};
static std::vector<std::string> SUPPORT_OP_NAME = {__SUPPORT_OP_NAME};

}


#endif
