#include <iostream>
#include "src/kernel.cuh"
#include <iomanip>
#include <fstream>

void WriteMatrixToFile(const char* filename, const float* matrix, int rows, int cols) {
    std::ofstream outFile(filename);

    // ����ļ��Ƿ�ɹ���
    if (!outFile.is_open()) {
        std::cerr << "�޷����ļ�: " << filename << std::endl;
        return;
    }

    // д���������
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[i * cols + j];  // д�����Ԫ��
            if (j < cols - 1) {
                outFile << " ";  // ÿ��Ԫ��֮��ӿո�
            }
        }
        outFile << "\n";  // ÿ�н�������
    }

    outFile.close();  // �ر��ļ�
    std::cout << "������д���ļ�: " << filename << std::endl;
}

//width*width�ľ������
void MatrixMultiplyOnDevice(float* hostP,const float* hostM, const float* hostN,const int width)
{
    //����һ��������ڴ��С
    int sizeInBytes = width*width*sizeof(float);
    float *devP,*devM,*devN;

    //�����Դ�
    cudaMalloc((void**)&devP,sizeInBytes);
    cudaMalloc((void**)&devM,sizeInBytes);
    cudaMalloc((void**)&devN,sizeInBytes);

    //��M��N����ֵ����GPU
    cudaMemcpy(devM,hostM,sizeInBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(devN,hostN,sizeInBytes,cudaMemcpyHostToDevice);

    //����blockΪ1���߳�Ϊ2D������Ϊwidth*width����˲��ܴ���32��32*32=1024��,���򷵻�0��
    dim3 threads(width,width);
    dim3 blocks(1,1);
    MatrixMultiplyKernel<<<blocks,threads>>>(devP,devM,devN,width);

    //ȡ�ؽ��
    cudaMemcpy(hostP,devP,sizeInBytes,cudaMemcpyDeviceToHost);

    //�ͷ��ڴ�
    cudaFree(devP);
    cudaFree(devM);
    cudaFree(devN);
}

void GenerateRandomMatrix(float* matrix, int width)
{
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < width * width; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // 0 �� 10 ֮������������
    }
}

// ��ӡ����
void PrintMatrix(const float* matrix, int width, const std::string& name)
{
    std::cout << name << ":\n";
    // ���ø�������ʽ���̶���ʽ������1λС��
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

    // ��ѡ���ָ�Ĭ�ϸ�ʽ����Ӱ�����������
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);  // �ָ�Ĭ��6λ��Ч����
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
