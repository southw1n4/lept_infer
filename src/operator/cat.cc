#include "operator/cat.h"
#include "basic/tools.h"

namespace leptinfer{

void _cat_helper(std::vector<int>& shape1, std::vector<int>& shape, std::vector<int>& idx, int p, const Tensor& a, const Tensor& b, Tensor& c, int dim){
    if(idx.size() == shape.size()) {
        int t = idx[dim];
        if(t >= shape1[dim]) {
            idx[dim] -= shape1[dim];
            c(idx) = b(idx);
            idx[dim] += shape1[dim];
        }else {
            c(idx) = a(idx);
        }
    }

    for(int i = 0; i < shape[p]; ++ i) {
        idx.push_back(i);
        _cat_helper(shape1, shape,  idx, p + 1, a, b, c,dim);
        idx.pop_back();
    }

}
 
Tensor Cat::operator()(const Tensor& a){
    auto y1 = a;
    auto y2 = *in[0];
    std::vector<int> shape1 = y1.shape(); 
    std::vector<int> shape2 = y2.shape();
    for(int i = 0; i < shape2.size(); ++ i){
        if(shape1[i] != shape2[i] && i != dim_){
            ERROR("cant not cat fort this two tensor!\n");
        }
    }

    auto shape = shape1;
    shape[dim_] = shape1[dim_] + shape2[dim_];

    auto y = Tensor(shape);
    std::vector<int> idx;
    _cat_helper(shape1, shape, idx, 0, y1, y2, y, dim_);

    return y;
}

void Cat::forward(){
    auto y = (*this)(*in[1]);
    notify(std::make_shared<Tensor>(y));
}

}

