#include "onnxparser/parser_helper.h"

#include "basic/tools.h"
#include "operator/conv.h"

namespace leptinfer{
Op* parse_conv(::onnx::NodeProto& node,
                std::unordered_map<std::string, Tensor*>& named_tensors,
                std::unordered_map<std::string, Op*>& named_ops){

    std::vector<int> dilations, kernel_shape, pads, strides;
    int group;

    std::unordered_map<std::string, int> attribute;
    for(int i = 0; i < node.attribute_size(); ++ i) {
        auto& attr = node.attribute(i);
        auto& name = attr.name();

        switch(name[0]) {
            case 'd': {
                    for(int j = 0; j < attr.ints_size(); ++ j) dilations.push_back(attr.ints(j));
                    break;
                }
            case 'g':{
                    group = attr.i();
                    break;
                }
            case 'p':{
                    for(int j = 0; j < attr.ints_size(); ++ j) pads.push_back(attr.ints(j));
                    break;
                }
            case 's':{
                    for(int j = 0; j < attr.ints_size(); ++ j) strides.push_back(attr.ints(j));
                    break;

                }
            case 'k':{
                    for(int j = 0; j < attr.ints_size(); ++ j) kernel_shape.push_back(attr.ints(j));
                    break;
                }
            default:
                ERROR("unsupported attribute: %s", name.c_str());
        }
    }

    std::string x = node.input(0);
    std::string w = node.input(1);
    std::string b = node.input(2);
    std::string y = node.output(0);
    int input_dims = named_tensors[w]->shape()[1];
    int output_dims = named_tensors[w]->shape()[0];

    Conv2d* next_op = new Conv2d(input_dims, output_dims, kernel_shape, strides, pads, dilations, group);
    next_op->set_weight(named_tensors[w]);
    next_op->set_bias(named_tensors[b]);

    for(auto it = named_ops.begin(); it != named_ops.end(); ++ it) {
        Op* prev_op = it->second;
        if(prev_op->output.count(x)) {
            prev_op->next.push_back(next_op);
        }
    }

    next_op->output[y] = true;
    return next_op;
}


}
