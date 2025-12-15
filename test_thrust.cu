#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
  thrust::device_vector<int> d_vec(10);
  thrust::sort(d_vec.begin(), d_vec.end());
  std::cout << "Thrust compiles!" << std::endl;
  return 0;
}
