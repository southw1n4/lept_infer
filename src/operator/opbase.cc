#include "operator/opbase.h"

#include <cmath>
#include <thread>

#include "basic/tools.h"


namespace leptinfer{

Op::~Op() {
    output.clear();
    for(auto& _t : in) {
        _t = NULL;
    }
}


void Op::notify(std::shared_ptr<Tensor> x) {
    status = true;

    if(is_output) {
        result = x;
    }
    for(auto& op : next_op) {
        op->in.push_back(x);
        op->wakeup();
    }
    for (int i = 0; i < in.size(); ++ i) {
        in[i] = NULL;
    }
    in.clear();
    x = NULL;
}

void Op::wakeup() {
    now_cnt --;
    if(now_cnt == 0) {
        now_cnt = all_cnt;
        std::thread run_op(run, this);
        run_op.detach();
    }
}

void Op::run(Op* op) {
    op->forward();
}

Tensor  gemm(const Tensor& a, const Tensor& b) {
    if(a.shape().size() != 2 || a.shape().back() != b.shape()[0]) {
        ERROR("dimension mismatch when gemm [%d, %d] and [%d, %d]\n", a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]);
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

void _base(std::vector<int>& shape, std::vector<int>& idx, int p, const Tensor& a, const Tensor& b, Tensor& c, int op = 0, int ndim = -1, int val = -1){
    if(idx.size() == shape.size()) {
        if(ndim != -1)  idx[ndim] = val;
        switch(op) {
            case 0: c(idx) = a(idx) + b(idx); break;
            case 1: c(idx) = a(idx) - b(idx); break;
            case 2: c(idx) = a(idx) * b(idx); break;
            case 3: c(idx) = a(idx) / b(idx); break;
        }
        return ;
    }

    for(int i = 0; i < shape[p]; ++ i) {
        idx.push_back(i);
        _base(shape, idx, p + 1, a, b, c, op, ndim, val);
        idx.pop_back();
    }

}

/*
 * +-/
 */
Tensor asmd_base(const Tensor& a, const Tensor& b, int op) {
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
    _base(sc, temp, 0, a, b, c, op);

    return c;

}
Tensor add(const Tensor& a, const Tensor& b) {
    return asmd_base(a, b, 0);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return asmd_base(a, b, 1);
}

Tensor div(const Tensor& a, const Tensor& b) {
    return asmd_base(a, b, 3);
}

Tensor mut(const Tensor& a, const Tensor& b) {
    return asmd_base(a, b, 2);
}

Tensor sum(const Tensor& a, int dim) {
    Tensor c(a);
    if(dim == -1) {
        float* ptr = (float*)c.data();
        float s = 0;
        for(int i = 0; i < c.size(); ++ i)
            s += ptr[i];
        c = Tensor(std::vector<int>(a.shape().size(), 1), {s});
    }else{
        auto shape = c.shape();
        int n = shape[dim];
        shape[dim] = 1;
        c = Tensor(shape);

        for(int i = 0; i < n; ++ i) {
            std::vector<int> idx;
            _base(shape, idx, 0, c, a, c, 0, dim, i);
        }
    }

    return c;
}



Tensor exp(const Tensor& a) {
    Tensor c(a);
    float* ptr = (float*)c.data();
    for(int i = 0; i < c.size(); ++ i)
        ptr[i] = std::exp(ptr[i]);

    return c;
}

Tensor sqrt(const Tensor& a) {
    Tensor c(a);
    float* ptr = (float*)c.data();
    for(int i = 0; i < c.size(); ++ i)
        ptr[i] = std::sqrt(ptr[i]);

    return c;
}

Tensor tanh(const Tensor& a){

    Tensor c(a);
    float* ptr = (float*)c.data();
    for(int i = 0; i < c.size(); ++ i)
        ptr[i] = std::tanh(ptr[i]);

    return c;

}

} 
