#include <iostream>
#include "src/kernel.cuh"
#include <iomanip>
#include <fstream>

void WriteMatrixToFile(const char* filename, const float* matrix, int rows, int cols) {
    std::ofstream outFile(filename);

    // 检查文件是否成功打开
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // 写入矩阵数据
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[i * cols + j];  // 写入矩阵元素
            if (j < cols - 1) {
                outFile << " ";  // 每个元素之间加空格
            }
        }
        outFile << "\n";  // 每行结束换行
    }

    outFile.close();  // 关闭文件
    std::cout << "矩阵已写入文件: " << filename << std::endl;
}

//width*width的矩阵相乘
void MatrixMultiplyOnDevice(float* hostP,const float* hostM, const float* hostN,const int width)
{
    //计算一个矩阵的内存大小
    int sizeInBytes = width*width*sizeof(float);
    float *devP,*devM,*devN;

    //分配显存
    cudaMalloc((void**)&devP,sizeInBytes);
    cudaMalloc((void**)&devM,sizeInBytes);
    cudaMalloc((void**)&devN,sizeInBytes);

    //将M和N矩阵值传到GPU
    cudaMemcpy(devM,hostM,sizeInBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(devN,hostN,sizeInBytes,cudaMemcpyHostToDevice);

    //设置block为1，线程为2D，数量为width*width，因此不能大于32（32*32=1024）,否则返回0；
    dim3 threads(width,width);
    dim3 blocks(1,1);
    MatrixMultiplyKernel<<<blocks,threads>>>(devP,devM,devN,width);

    //取回结果
    cudaMemcpy(hostP,devP,sizeInBytes,cudaMemcpyDeviceToHost);

    //释放内存
    cudaFree(devP);
    cudaFree(devM);
    cudaFree(devN);
}

void GenerateRandomMatrix(float* matrix, int width)
{
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < width * width; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // 0 到 10 之间的随机浮点数
    }
}

// 打印矩阵
void PrintMatrix(const float* matrix, int width, const std::string& name)
{
    std::cout << name << ":\n";
    // 设置浮点数格式：固定格式，保留1位小数
    std::cout << std::fixed << std::setprecision(2);

    for (int row = 0; row < width; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            std::cout << matrix[row * width + col] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // 可选：恢复默认格式（不影响其他输出）
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);  // 恢复默认6位有效数字
}

void MatrixMultiplyCPU(const float* A, const float* B, float* C, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

int main()
{
    int width = 32;
    float *hostM=new float[width * width];
    float *hostN=new float[width * width];
    float *hostP=new float[width * width];

    GenerateRandomMatrix(hostM,width);
    GenerateRandomMatrix(hostN,width);

    MatrixMultiplyOnDevice(hostP,hostM,hostN,width);

    // PrintMatrix(hostM,width,"hostM");
    // PrintMatrix(hostN,width,"hostN");
    PrintMatrix(hostP,width,"hostP");

    // WriteMatrixToFile("matrix_output.txt", hostP, width, width);

    // MatrixMultiplyCPU(hostM,hostN,hostP,width);
    // PrintMatrix(hostP,width,"CPU Calculated hostP");

    delete[] hostM;
    delete[] hostN;
    delete[] hostP;
    return 0;
}
