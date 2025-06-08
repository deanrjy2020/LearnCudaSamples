# LearnCudaSamples

Copied from /usr/local/cuda/samples/ on Jetson AGX Xavier with JetPack 5.1.5. Cuda version 11.4.19

最新的版本在这里: https://github.com/NVIDIA/cuda-samples/tree/master

make dbg=1 SMS=72
make clean dbg=1 SMS=72

Done with:
- 0_Simple
    - asyncAPI
        - (H2D async, kernel, D2H async), 用CPU记录时间和event记录GPU执行时间, 对比.
    - cppIntegration
        - C++ 包裹 CUDA 的简单封装，适合 plugin 或工程模块化使用。看看结构，了解怎么组织 .cu/.cpp 混编，不必深挖
    - cppOverload
        - cudaFuncGetAttributes(&attr, *func1);得到核函数编译后的信息, 如shared mem size, register num
        - 上面的是runtime时候知道, 如果在flags里面加--resource-usage或者--ptxas-options=-v, nvcc编译时候就能知道.
    - matrixMul
        - 核心：tile-based shared memory 实现 GEMM，是推理核心操作。重点掌握：BLOCKSIZE, shared memory tiling, loop unrolling, bank conflict 避免，和自己代码对比写一版.
        - port 到LearnAI/chapter2-cuda-programming/2.7-matmul-shared-memory with more comments.
    - matrixMulCUBLAS
        - 调用cublasSgemm做矩阵乘法,而不是自己写kernel函数. cublas是col-major, C++里面C=A*B要按照B*A传进去, 不用做任何转置. 具体见代码里面的comments.
    - simpleAssert
        - 可以在核函数里面用assert, 了解即可，可用在 debug 推理 kernel 的中间状态检查
        - C++里面也有对应的shell的uname函数, 得到系统的信息.
    - simpleOccupancy
        - 由cudaOccupancyMaxPotentialBlockSize()推荐block size, 然后用cudaOccupancyMaxActiveBlocksPerMultiprocessor()算得occupancy. 重点理解 occupancy 概念 + 用这个 API 做参数 sweep 自动调优. 见cuda hw page warp部分.
    - simplePrintf
        - 核函数内 printf
    - simpleStreams
        - 单单memcopy时间7.89, 单单跑kernel时间10.29, 合在一起做10次求平均时间9.07, 分成4个stream并行处理时间5.33
    - simpleZeroCopy
        - 展示两种mapped内存分配方式, 1, 普通malloc, 然后cudaHostRegisterMapped注册实现map, 2, 用cudaHostAllocMapped实现map, 都是CPU端分配mem, 然后map后GPU可以直接用, 不用memcpy. 更多见cuda mem page.
    - vectorAdd
        - 非常简单的H2D, C=A+B, D2H
- 1_Utilities
    - deviceQuery
    - bandwidthTest
        - 就是测试H2D/D2H/D2D的速度. H2D 19.0 GB/s, D2H 19.0 GB/s, D2D 62.5 GB/s
        - Jetson AGX Xavier 使用统一内存（Unified Memory Architecture, UMA），意味着 CPU 和 GPU 都访问同一个 DRAM。这种设计避免了 discrete GPU 上 PCIe 复制的瓶颈. 但是"统一物理内存" ≠ "相同的访问效率". CPU 和 GPU 是通过不同的路径访问 DRAM, CPU 走 内存控制器(via ARM coherent interconnect, e.g., CCI/CCI-N), GPU 走 GPU DRAM interface（高带宽 memory controller, 可支持 HBM/LPDDR）. H2D/D2H是由 CPU 发起 copy 行为，DMA 机制或 CPU mem copy 都有限制在带宽和执行模型上. D2D是 在 GPU 上由 CUDA runtime 发起的 copy 操作，直接走 GPU 的内部 memory controller，属于最快路径.
- 6_Advanced
    - concurrentKernels
        - 8个kernel在8个stream上并发, 全部执行完后, 用一个sum核函数规约, 把前面8个执行的output (ticks)累加起来.
    - reduction
        - todo, next
