### A inference framework write by c++


### v1.0

#### start

+ test
```shell

git clone https://github.com/southw1n4/lept_infer
cd lept_infer
mkdir build && cd build
cmake ..
make test

```

+ benchmark

```shell
git clone https://github.com/southw1n4/lept_infer
cd lept_infer
mkdir build && cd build
cmake ..
make benchmark
```

#### use std::cout 

```c++

auto x = leptinfer::Tensor({3, 3, 3});
std::cout << x << std::endl;

/*here is the output
[[[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]]]

*/

```
