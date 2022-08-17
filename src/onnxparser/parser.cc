
#include "onnxparser/parser.h"

#include <fstream>

#include "basic/tools.h"
#include "onnxparser/onnx.proto3.pb.h"
#include "onnxparser/parser_helper.h"


namespace leptinfer{

Net* OnnxParser::parse(const std::string& file, bool varbose){


    varbose_ = varbose;
    net_ = NULL;

    ::onnx::ModelProto model_proto;
    std::ifstream input(file, std::ios::in | std::ios::binary); 
    if(!model_proto.ParseFromIstream(&input)) {
        ERROR("failed to parse onnx file\n");
        return NULL;
    }

    INFO("String parseing network model\n");
    INFO("------------------------------------------------------\n");
    INFO("Input filename:\t %s\n", file.c_str());
    INFO("ONNX IR version:\t %ld\n", model_proto.ir_version());
    INFO("Producer name:\t %s\n", model_proto.producer_name().c_str());
    INFO("Producer version:\t %s\n", model_proto.producer_version().c_str());
    INFO("Model version:\t %ld\n", model_proto.model_version());
    INFO("------------------------------------------------------\n");


    auto graph = model_proto.graph();

    net_ = new Net();
    if(!parser_tensor(graph)){
        delete  net_;
        net_ = NULL;

        ERROR("failed parse network model\n");
        return NULL;
    }

    if(!parser_op(graph)){
        delete  net_;
        net_ = NULL;

        ERROR("failed parse network model\n");
        return NULL;
    }

    INFO("finish parsing network model\n");

    net_->top_sort(net_->ops);


    return net_;
}

bool OnnxParser::parser_tensor(::onnx::GraphProto& grpgh) {

    return true;
}

bool OnnxParser::parser_op(::onnx::GraphProto& grpgh) {
    for(int i = 0; i < grpgh.node_size(); ++ i) {
        auto node = grpgh.node(i);
        auto node_name = node.name();

        if(varbose_) {
            INFO("parsing layer: %s\n", node_name.c_str());
        }

        const std::string op_name = node.op_type();
        if(op_name != "Conv") continue;

        auto it_name = std::find(SUPPORT_OP_NAME.begin(), SUPPORT_OP_NAME.end(), op_name);
        if(it_name == SUPPORT_OP_NAME.end()) {
            ERROR("unsupported opeartor: %s", op_name.c_str());

            return false;
        }

        auto it_func = SUPPROT_OP_FUNC.begin() + (it_name - SUPPORT_OP_NAME.begin());
        auto op = (*it_func)(node, named_tensors_, named_ops_);

        named_ops_[node_name] = op;
        
    }

    return true;
}
}
