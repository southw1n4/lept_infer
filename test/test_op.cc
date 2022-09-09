
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <numeric>
#include <iostream>
#include <cstring>

#include "operator/linear.h"
#include "operator/activate.h"
#include "operator/conv.h"
#include "operator/pool.h"
#include "operator/norm.h"

#include "basic/tensor.h"

static int op_cnt = 0;
static int op_true = 0;
static int n_point = 53;

using namespace leptinfer;

#define OUT(num, status) \
    do{\
        for(int i = 0; i < (num); ++ i) printf("%c", '.');\
        printf("[%s]\n", #status);\
    }while(0)

#define TEST(func) \
    do{\
        int n_out_point = n_point - strlen(#func);\
        op_cnt ++;\
        printf("%s", #func);\
        if(func()) {\
            op_true++;\
            OUT(n_out_point, OK); \
        }\
    else{\
            n_out_point -= 4;\
            OUT(n_out_point, FAILED); \
        }\
    }while(0)


Tensor random_tenosr(std::vector<int> shape, int len = 2) {
    if(shape.size() == 0) {
        for(int i = 0; i < len; ++ i) shape.push_back(rand() % 100 + 1);
    }
    std::vector<float> nums;
    int size =  std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    for(int i = 0; i < size; ++ i) nums.push_back((float)(rand() % 10) / 5);

    return Tensor(shape, nums);
}
bool test_linear(){
    auto op = leptinfer::Linear(2, 3);
    auto x = Tensor({3, 2}, {1, 2, 3, 4, 5, 6});
    op.set_weight(new Tensor({2, 3}, {6, 5, 4, 3 ,2, 1}));
    op.set_bias(new Tensor({1, 3}, {0.4, 0.5, 0.6}));

    return op(x) == Tensor({3, 3}, {12.4, 9.5, 6.6,
                                    30.4, 23.5, 16.6,
                                    48.4, 37.5, 26.6});
}

bool test_softmax() {

    auto op = leptinfer::Softmax(0);
    auto x = Tensor({3, 2}, {1, 2, 3, 4, 5, 6});
    auto y = op(x);
    bool flag1 = (op(x) == Tensor({3, 2}, {0.0159, 0.0159,
                                           0.1173, 0.1173,
                                           0.8668, 0.8668,}));
    op = leptinfer::Softmax(1);
    y = op(x);

    bool flag2 = (op(x) == Tensor({3, 2}, {0.2689, 0.7311,
                                           0.2689, 0.7311,
                                           0.2689, 0.7311}));

    return flag1 && flag2;
}



bool test_tanh() {

    std::vector<float> t = {1, 2, 3, 4, 5, 6};

    auto op = TanH();
    auto x = Tensor({3, 2}, t);
    for(int i = 0; i < t.size(); ++ i) t[i] = std::tanh(t[i]);
    auto y = Tensor({3, 2}, t);

    return op(x) == y;

}

bool test_conv2d() {
    auto x = Tensor({2, 3, 3, 3}, Tensor::tensor_type::TYPE_FP32, 1);
    auto w = new Tensor({3, 3, 3, 3}, Tensor::tensor_type::TYPE_FP32, 1);
    auto b = new Tensor({3}, {0, 1 ,2});
    auto z1 = Tensor({2, 3, 1, 1}, Tensor::tensor_type::TYPE_FP32, 27);
    auto z2 = Tensor({2, 3, 3, 3}, {12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12});

    auto z3 = Tensor({2, 3, 3, 3}, {12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     13, 19, 13, 19, 28, 19, 13, 19, 13,
                                     14, 20, 14, 20, 29, 20, 14, 20, 14,
                                     12, 18, 12, 18, 27, 18, 12, 18, 12,
                                     13, 19, 13, 19, 28, 19, 13, 19, 13,
                                     14, 20, 14, 20, 29, 20, 14, 20, 14});
                                     

    auto op1 = Conv2d(3, 3, {3}, {1}, {0});
    op1.set_weight(new Tensor(*w));
    bool flag1 = (z1 == op1(x));

    auto op2 = Conv2d(3, 3, {3}, {1}, {1});
    op2.set_weight(new Tensor(*w));
    bool flag2 = (z2 == op2(x));

    auto op3 = Conv2d(3, 3, {3}, {1}, {1}, {0}, 0, true);
    op3.set_weight(new Tensor(*w));
    op3.set_bias(new Tensor(*b));
    bool flag3 = (z3 == op3(x));

    delete w;
    delete b;

    return flag1 && flag2 && flag3;
}

bool test_maxpool2d() {
    auto x = Tensor({2, 2, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
                                  1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,});

    auto z1 = Tensor({2, 2, 2, 2}, {6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8});
    auto z2 = Tensor({2, 2, 3, 3}, {1, 3, 4, 5, 7, 8, 5, 7, 8,
                                    1, 3, 4, 5, 7, 8, 5, 7, 8,
                                    1, 3, 4, 5, 7, 8, 5, 7, 8,
                                    1, 3, 4, 5, 7, 8, 5, 7, 8
                                    });
    auto op = MaxPool2d(2, 2, 0);
    bool flag1 = op(x) == z1;

    op = MaxPool2d(2, 2, 1);
    bool flag2 = op(x) == z2;
    
    return flag2 && flag1;
}

bool test_relu() {
    auto x = Tensor({2, 2}, {-1, 3, 0, -2});
    auto op = ReLU();
    auto z = Tensor({2, 2}, {0, 3, 0, 0});

    auto y = Tensor({3, 3, 3, 3}); 

    return op(x) == z;
}

bool test_batchnrom2d() {
    auto x = Tensor({2, 3, 2, 2}, {1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12,
                                   1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12});
    auto z = Tensor({2, 3, 2, 2},{-1.0588, -0.1863, 0.6863,  1.5588, -0.6588,  0.2137, 1.0863,  1.9588, -0.2588,  0.6137, 1.4863,  2.3588,
                                  -1.0588, -0.1863, 0.6863,  1.5588, -0.6588,  0.2137, 1.0863,  1.9588, -0.2588,  0.6137, 1.4863,  2.3588});

    auto op = BatchNorm2d(3);
    auto gamma = Tensor({1, 3, 1, 1}, {0.25, 0.65, 1.05});
    auto beta = Tensor({1, 3, 1, 1}, {1.0429, 1.0429, 1.0429});
    op.set_gamma(new Tensor(gamma));
    op.set_beta(new Tensor(beta));

    return op(x) == z;

}




void test_op() {
    printf("=========================TEST_OP=========================\n");
    TEST(test_linear);
    TEST(test_softmax);
    TEST(test_tanh);
    TEST(test_conv2d);
    TEST(test_maxpool2d);
    TEST(test_relu);
    TEST(test_batchnrom2d);
    printf("PASS: %d/%d\n",op_true, op_cnt);
    printf("=========================TEST_OP=========================\n");
}


