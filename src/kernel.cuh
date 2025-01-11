//
// Created by 90681 on 2025/1/11.
//

#ifndef KERNEL_CUH
#define KERNEL_CUH


__global__ void MatrixMultiplyKernel(float* hostP,const float* hostM, const float* hostN,const int width);


#endif //KERNEL_CUH
