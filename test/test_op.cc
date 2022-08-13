#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <numeric>

#include "operator/linear.h"
#include "operator/activate.h"

static int op_cnt = 0;
static int op_true = 0;

using namespace leptinfer;
#define TEST(func) \
    do{\
        op_cnt ++;\
        if(func()) {\
            op_true++;\
            printf("%s\t.....................................[OK]\n", #func); \
        }\
        else{\
            printf("%s\t.................................[FAILED]\n", #func); \
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
    op.set_weight(Tensor({2, 3}, {6, 5, 4, 3 ,2, 1}));
    op.set_bias(Tensor({1, 3}, {0.4, 0.5, 0.6}));

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



void test_op() {
    printf("=========================TEST_OP=========================\n");
    TEST(test_linear);
    TEST(test_softmax);
    TEST(test_tanh);
    printf("PASS:\t%d\nALL:\t%d\n",op_true, op_cnt);
    printf("=========================TEST_OP=========================\n");

}


