#include "onnxparser/parser_helper.h"

#include "basic/tools.h"
#include "operator/conv.h"
#include "operator/activate.h"
#include "operator/linear.h"
#include "operator/flatten.h"

#define PARSE(name) \
        Op* parse_##name(::onnx::NodeProto& node, \
                std::unordered_map<std::string, Tensor*>& named_tensors)

namespace leptinfer{
PARSE(conv){
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

    std::string w = node.input(1);
    std::string b = node.input(2);
    int input_dims = named_tensors[w]->shape()[1];
    int output_dims = named_tensors[w]->shape()[0];

    Conv2d* next_op = new Conv2d(input_dims, output_dims, kernel_shape, strides, pads, dilations, group);
    next_op->set_weight(named_tensors[w]);
    next_op->set_bias(named_tensors[b]);

    return next_op;
}

PARSE(relu) {

    return new ReLU();
}

PARSE(gemm) {
    std::string w = node.input(1);
    std::string b = node.input(2);

    int input_dims = named_tensors[w]->shape()[1];
    int output_dims = named_tensors[w]->shape()[0];

    Linear* next_op = new Linear(input_dims, output_dims, true);
    Tensor new_w(named_tensors[w]->T());
    named_tensors[b]->reshape({1, output_dims});
    delete named_tensors[w];

    next_op->set_weight(new Tensor(new_w));
    next_op->set_bias(named_tensors[b]);

    return next_op;
}


PARSE(flatten) {

    return new Flatten();
}

}
