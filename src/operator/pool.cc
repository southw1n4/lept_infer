
#include "operator/pool.h"
#include "basic/tools.h"


namespace leptinfer{
MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding):
    kernel_size_(kernel_size), stride_(stride), padding_(padding){now_cnt =all_cnt = 1;}


Tensor MaxPool2d::operator()(const Tensor& x) {


    std::vector<int> sx = x.shape();
    if(sx.size() != 4) {
        ERROR("cant run conv2d in this shape, please check!");
    }

    int B  = sx[0];
    int C  = sx[1];
    int sh = sx[2];
    int sw = sx[3];
    int eh = (2 * padding_ + sh - kernel_size_) / stride_ + 1;
    int ew = (2 * padding_ + sw - kernel_size_) / stride_ + 1;
    Tensor y = Tensor({B, C,  eh, ew});

#ifdef HAND

    for(int batch = 0; batch < B; ++ batch) {

        for(int channel = 0; channel < C; ++ channel) {

            for(int py = 0; py < eh; ++ py) {
                for(int px = 0; px < ew; ++ px) {

                    float* x_ptr = (float *)x.data() + (batch * C + channel) * sh * sw;
                    float* y_ptr = (float *)y.data() + (batch * C + channel) * eh * ew + py * ew + px; 
                    compute_local(py, px, y_ptr, x_ptr, sw, sh);
                }
            }
        }
    }

#endif
    return y;
}

void MaxPool2d::compute_local(int tgt_ycoord, int tgt_xcoord, float* tgt_ptr, float* src_ptr, int src_w, int src_h){

#ifdef HAND
    int src_ycoord = tgt_ycoord * stride_ - padding_; 
    int src_xcoord = tgt_xcoord * stride_ - padding_; 
    *tgt_ptr = -1e16;

    for(int dy = 0; dy < kernel_size_; ++ dy) {
        for(int dx = 0; dx < kernel_size_; ++ dx) {
            int ty = src_ycoord + dy;
            int tx = src_xcoord + dx;

            float v = (ty < 0 || tx < 0 || ty >= src_h || tx >= src_w) ? 0: src_ptr[ty * src_w + tx];
            *tgt_ptr = std::max(*tgt_ptr, v);

        }
    }
#endif
}


void MaxPool2d::forward() {

    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}
}
