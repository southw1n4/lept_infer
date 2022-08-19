#include "operator/conv.h"

#include "basic/tools.h"


namespace leptinfer{

Conv2d::Conv2d(int input_dims, int output_dims, 
        std::vector<int> kernel_size,
        std::vector<int> stride,
        std::vector<int> pads,
        std::vector<int> dilations,
        int group,
        bool required_bias){
    input_dims_  = input_dims;
    output_dims_ = output_dims;
    kernel_size_ = kernel_size[0];
    stride_      = stride[0];
    padding_     = pads[0];
    all_cnt = now_cnt = 1;
}

void Conv2d::forward() {

    auto y = (*this)(*in[0]);
    notify(std::make_shared<Tensor>(y));
}

Conv2d::~Conv2d() {
    if(weight_ != NULL) delete weight_;
    if(bias_ != NULL) delete bias_;
}


Tensor Conv2d::operator()(const Tensor& x){
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
   Tensor y = Tensor({B, output_dims_,  eh, ew});

#ifdef HAND

   float* bw_ptr = (float *) weight_->data();
   for(int batch = 0; batch < B; ++ batch) {
       float* bx_ptr = (float *)x.data() + batch * C * sh * sw; 
       float* by_ptr = (float *)y.data() + batch * output_dims_ * eh * ew;

       for(int out_c = 0; out_c < output_dims_; ++ out_c) {
           float* cw_ptr = bw_ptr + input_dims_ * kernel_size_ * kernel_size_;
           float* cy_ptr = by_ptr + out_c * eh * ew;
           float bias = (required_bias_ ? *((float *)bias_->data() + out_c) : 0.f); 

           for(int ph = 0; ph < eh; ++ ph) {
               for(int pw = 0; pw < ew; ++ pw) {

                   float* y_ptr = cy_ptr + ph * ew + pw;
                   float* w_ptr = cw_ptr; 
                   float* x_ptr = bx_ptr;
                   const int x_offset = sh * sw;
                   const int w_offset = kernel_size_ * kernel_size_;
                   compute_local(ph, pw, x_ptr, y_ptr, w_ptr, x_offset, w_offset, sw, sh, bias);
               }
           }
       }
    
   }
#endif


    return y;
}

void Conv2d::compute_local(int tgt_xcoord, int tgt_ycoord, float* src_ptr, float* tgt_ptr, float* weight_ptr, 
                           int src_offset, int weight_offset, int src_w, int src_h, float bias){

#ifdef HAND
    *tgt_ptr = bias;
    int src_xcoord = tgt_xcoord * stride_ - padding_; 
    int src_ycoord = tgt_ycoord * stride_ - padding_; 

    for(int channel = 0; channel < input_dims_; ++ channel) {

        float single_channel_sum = 0.f;
        for(int dy = 0; dy < kernel_size_; ++ dy) {
            for(int dx = 0; dx < kernel_size_; ++ dx) {

                float v1 = weight_ptr[dy * kernel_size_ + dx];
                int ty = src_ycoord + dy;
                int tx = src_xcoord + dx;

                float v2 = (ty < 0 || tx < 0 || ty >= src_h || tx >= src_w) ? 0: src_ptr[ty * src_w + tx];
                single_channel_sum += v1 * v2;
            }
        }

        src_ptr += src_offset;
        weight_ptr += weight_offset;
        *tgt_ptr += single_channel_sum;
    }
#endif
    
}


void Conv2d::set_weight(Tensor* a){
    if(weight_ != NULL) delete weight_;
    weight_ = a;

}

void Conv2d::set_bias(Tensor* a){
    required_bias_ = true;
    if(bias_ != NULL) delete bias_;
    bias_ = a;
}
}
