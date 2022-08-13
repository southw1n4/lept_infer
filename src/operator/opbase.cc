#include "operator/opbase.h"
#include "basic/tools.h"


namespace leptinfer{

Tensor  gemm(const Tensor& a, const Tensor& b) {
    if(a.shape().size() != 2 || a.shape().back() != b.shape()[0]) {
        ERROR("dimension mismatch");
    }
    auto shape1 = a.shape(), shape2 = b.shape();
    int na = shape1[0], ma = shape1[1];
    int nb = shape2[0], mb = shape2[1];
    auto c = Tensor({na, mb});

#ifdef HAND

    float* pa = (float *) a.data(); 
    float* pb = (float *) b.data();
    float* pc = (float *) c.data();
    for(int i = 0; i < na; ++ i) {
        for(int j = 0; j < mb; ++ j) {
            float& c = *(pc + i * mb + j);
            for(int k = 0; k < ma; ++ k) {
                c += *(pa + i * ma + k) * *(pb + k * mb + j) ;
            }

        }
    }


#endif

    return c;

}

void _add(std::vector<int>& shape, std::vector<int>& idx, int p, const Tensor& a, const Tensor& b, Tensor& c){
    if(idx.size() == shape.size()) {
        c(idx) = a(idx) + b(idx);
        return ;
    }

    for(int i = 0; i < shape[p]; ++ i) {
        idx.push_back(i);
        _add(shape, idx, p + 1, a, b, c);
        idx.pop_back();
    }

}
Tensor add(const Tensor& a, const Tensor& b) {
    if(a.shape().size() != b.shape().size()) {
        ERROR("dimension mismatch");
    }
    std::vector<int> sa = a.shape();
    std::vector<int> sb = b.shape();
    std::vector<int> sc;
    for(int i = 0; i < sa.size(); ++ i) {
        sc.push_back(std::max(sa[i], sb[i]));
    }

    Tensor c(sc);
    std::vector<int> temp;
    _add(sc, temp, 0, a, b, c);

    return c;
}
} 
