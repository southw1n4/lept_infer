
#include "onnxparser/parser.h"

#include <basic/tensor.h>
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

    input.close();
    INFO("finish parsing network model\n");

    return net_;
}

bool OnnxParser::parser_tensor(::onnx::GraphProto& grpgh) {

    for(int i = 0; i < grpgh.initializer_size(); ++ i) {
        auto& tensor = grpgh.initializer(i);
        auto& tensor_name = tensor.name();
        std::vector<int> shapes;

        for(int j = 0; j < tensor.dims_size(); ++ j) shapes.push_back(tensor.dims(j));
        Tensor* _temp_tensor = new Tensor(shapes);

        if(_temp_tensor == NULL || _temp_tensor->data() == NULL) {
            ERROR("wrong when create a tensor!!");
            return false;
        }
        memcpy(_temp_tensor->data(), tensor.raw_data().data(), _temp_tensor->size() * sizeof(float));

        named_tensors_[tensor_name] = _temp_tensor;
    }

    return true;
}

bool OnnxParser::parser_op(::onnx::GraphProto& grpgh) {

    for(int i = 0; i < grpgh.output_size(); ++ i) {
        auto output_name = grpgh.output(i).name();
        output_name_[output_name] = true;
    }

    for(int i = 0; i < grpgh.node_size(); ++ i) {
        auto node = grpgh.node(i);
        auto node_name = node.name();

        if(varbose_) {
            INFO("parsing layer: %s\n", node_name.c_str());
        }

        const std::string op_name = node.op_type();

        auto it_name = std::find(SUPPORT_OP_NAME.begin(), SUPPORT_OP_NAME.end(), op_name);
        if(it_name == SUPPORT_OP_NAME.end()) {
            ERROR("unsupported opeartor: %s", op_name.c_str());

            return false;
        }

        auto it_func = SUPPROT_OP_FUNC.begin() + (it_name - SUPPORT_OP_NAME.begin());
        auto op = (*it_func)(node, named_tensors_);

        for(int j = 0; j < node.input_size(); ++ j) {
            auto input_name = node.input(j);
            if(named_tensors_.count(input_name)) continue;

            for(auto it_ops = named_ops_.begin(); it_ops != named_ops_.end(); ++ it_ops) {
                if(it_ops->second->output.count(input_name)) {
                    it_ops->second->next_op.push_back(op);
                }
            }
        }

        for(int j = 0; j < node.output_size(); ++ j) {
            auto output_name = node.output(j);
            op->output[output_name] = true;

            if(output_name_.count(output_name)) {
                op->is_output = true;
                net_->output_idx.push_back(i);
            }
        }

        named_ops_[node_name] = op;
        net_->ops.push_back(op);
    }

    return true;
}
}
