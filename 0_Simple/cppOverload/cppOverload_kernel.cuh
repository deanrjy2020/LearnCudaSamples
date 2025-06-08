__global__ void simple_kernel(const int *pIn, int *pOut, int a)
{
    // 一个block用了256*4=1024 Byte smem, 就是放256 int
    __shared__ int sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    // 每个thread负责一个int, 从gmem copy到smem
    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    // 随便处理一下, 写回gmem
    pOut[tid] = sData[threadIdx.x]*a + tid;;
}

__global__ void simple_kernel(const int2 *pIn, int *pOut, int a)
{
    // 256* 2 int * 4 Byte = 2048B
    __shared__ int2 sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = (sData[threadIdx.x].x + sData[threadIdx.x].y)*a + tid;;
}

__global__ void simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a)
{
    // 1024B + 1024B
    __shared__ int sData1[THREAD_N];
    __shared__ int sData2[THREAD_N];
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    sData1[threadIdx.x] = pIn1[tid];
    sData2[threadIdx.x] = pIn2[tid];
    __syncthreads();

    pOut[tid] = (sData1[threadIdx.x] + sData2[threadIdx.x])*a + tid;
}
