//
// Created by 90681 on 2025/1/11.
//

#include "kernel.cuh"


__global__ void MatrixMultiplyKernel(float* hostP,const float* hostM, const float* hostN,const int width)
{
    //tx、ty：flat的线程id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float pValue = 0;

    for(int i =0;i<width;i++) {
        float m =hostM[tx*width+i];
        float n =hostN[i*width+ty];
        pValue += m*n;
    }
    hostP[tx*width+ty] = pValue;
}