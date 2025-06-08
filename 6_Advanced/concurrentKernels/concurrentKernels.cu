/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//
// This sample demonstrates the use of streams for concurrent execution. It also
// illustrates how to introduce dependencies between CUDA streams with the
// cudaStreamWaitEvent function.
//

// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <helper_functions.h>

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
__global__ void clock_block(clock_t *d_o, clock_t clock_count) {
  unsigned int start_clock = (unsigned int)clock();

  clock_t clock_offset = 0;

  // 运行够clock_count个时钟周期, 即这么多ticks就可以退出了.
  // 核函数的output就是把这个clock_offset (应该略大于clock_count)输出
  while (clock_offset < clock_count) {
    unsigned int end_clock = (unsigned int)clock();

    // The code below should work like
    // this (thanks to modular arithmetics):
    //
    // clock_offset = (clock_t) (end_clock > start_clock ?
    //                           end_clock - start_clock :
    //                           end_clock + (0xffffffffu - start_clock));
    //
    // Indeed, let m = 2^32 then
    // end - start = end + m - start (mod m).

    clock_offset = (clock_t)(end_clock - start_clock);
  }

  d_o[0] = clock_offset;
}

// Single warp reduction kernel
// N=8, d_clocks就是一个长度为8个array, 放着8个clocks, 即前面8个kernel的执行时间, 单位tick
// 把这些时间加总起来，写回 d_clocks[0]
__global__ void sum(clock_t *d_clocks, int N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // 分配了一个大小为 32 的共享内存数组，用于归约中间结果
  // 这里假设最多只有 32 个线程（即一个 warp），因为这个是 single-warp reduction
  __shared__ clock_t s_clocks[32];

  clock_t my_sum = 0;

  // thread 0 处理 i=0, i=32, i=64
  // N=8, 只有thread 0~7在工作, 从global mem读入, 写到自己的register my_sum里面(单单copy).
  // 其他thread无用(对应的my_sum=0).
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    my_sum += d_clocks[i];
  }

  // 每个thread把自己的my_sum写到对应的smem位置.
  s_clocks[threadIdx.x] = my_sum;
  // 等待所有线程完成写入
  cg::sync(cta);

  // 归约（Reduction）逻辑
  // 在smem里面完成规约.
  // 对0~31共32个thread, i=16, 8, 4, 2, 1
  for (int i = 16; i > 0; i /= 2) {
    // i=16, 0~15号thread才能进来, 16~31号thread跳过.
    if (threadIdx.x < i) {
      // 0~15号thread的i=16 (第一次loop), 把smem的后半部分加到前半部分.
      s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
    }

    // 每次都是32个thread都做完了在进行下一次loop
    cg::sync(cta);

    // loop2, i=8, 32个thread都有这个loop, 但是只有0~7号thread才能进if
    //    进去后把4~7号的4个加到0~3上来
    // loop3, i=4, 32个thread都有这个loop, 但是只有0~3号thread才能进if
    //    进去后把2~3号的2个加到0~1上来
    // loop4, i=2, 32个thread都有这个loop, 但是只有0~1号thread才能进if
    //    进去后把1号的1个加到0上来
    // 最终32个smem里面的累加在s_clock[0]上
  }

  // 从smem写回gmem
  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  int nkernels = 8;             // number of concurrent kernels
  int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(clock_t);  // number of data bytes
  // 让每个kernel跑10ms
  float kernel_time = 10;                   // time the kernel should run in ms
  float elapsed_time;                       // timing variables
  int cuda_device = 0;

  printf("[%s] - Starting...\n", argv[0]);

  // get number of kernels if overridden on the command line
  if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
    nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
    nstreams = nkernels + 1;
  }

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  cuda_device = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDevice(&cuda_device));

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

  if ((deviceProp.concurrentKernels == 0)) {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

  // allocate host memory
  clock_t *a = 0;  // pointer to the array data in host memory
  checkCudaErrors(cudaMallocHost((void **)&a, nbytes));

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

  // allocate and initialize an array of stream handles
  cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  for (int i = 0; i < nstreams; i++) {
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }

  // create CUDA event handles
  // 用来测量总的 GPU 执行时间, 被记录在stream 0上
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  // 8个kernel对应8个event
  cudaEvent_t *kernelEvent;
  kernelEvent = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));

  // 使用 cudaEventDisableTiming 表示这些事件只是用于同步，不记录时间（避免引入隐式同步，影响并发性）
  for (int i = 0; i < nkernels; i++) {
    checkCudaErrors(
        cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
  }

  //////////////////////////////////////////////////////////////////////
  // time execution with nkernels streams
  // kernel_time 是期望 kernel 运行的毫秒数, 10ms
  // deviceProp.clockRate 是 GPU 的时钟频率（单位 kHz）, 即1377000 kHZ, 是每秒有多少个时钟周期,
  // 即1377000000 ticks/s
  // 两个相乘表示运行10ms需要的ticks个数 = 10ms * 1/1000 * 1377000000 = 10 * 1377000 = 13770000 ticks
  // 这个就是核函数里面的time_count.
  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to
  // prevent hangs reduce time_clocks.
  // 这里arm架构改成和x86架构一样, 因为后面的打印信息没考虑到这个100.
  //clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif

  cudaEventRecord(start_event, 0);

  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    // 8个核函数在8个stream上跑, 并行, 一个kernel只有一个thread
    clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
    total_clocks += time_clocks;
    // 在每个核函数后面都放一个event
    checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));

    // make the last stream wait for the kernel event to be recorded
    // stream 9个, 0~8, kernel 8个, 0~7
    // 最后一个stream (stream 8)等着前面0~7个kernel, 它会等待所有 kernel 执行完再继续执行汇总操作
    checkCudaErrors(
        cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0));
  }

  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  // 32个thread, sum kernel 负责汇总多个 d_a[i] 的数据, 然后用异步拷贝从 device 到 host.
  // 两个操作都在 streams 8上执行，而这个 stream 8 又等待了所有 kernel event，
  // 因此它们会在所有 kernel 执行完之后才开始.
  sum<<<1, 32, 0, streams[nstreams - 1]>>>(d_a, nkernels);
  checkCudaErrors(cudaMemcpyAsync(
      a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams - 1]));

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // in this sample we just wait until the GPU is done
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

  // expected 信息, 每个kernel跑10ms, 共8个, 80ms=0.08s
  printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
         nkernels * kernel_time / 1000.0f);
  // 理论完全并行, 8个和1个时间一样, 0.01s
  printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
         nkernels, kernel_time / 1000.0f);
  printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] > total_clocks);
  printf("Measured tick = %d, total_clocks = %d\n",
    (unsigned int) a[0], (unsigned int)total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++) {
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(kernelEvent[i]);
  }

  free(streams);
  free(kernelEvent);

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFreeHost(a);
  cudaFree(d_a);

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}

/*

[./concurrentKernels] - Starting...
GPU Device 0: "Xavier" with compute capability 7.2

> Detected Compute SM 7.2 hardware with 8 multi-processors
Expected time for serial execution of 8 kernels = 0.080s
Expected time for concurrent execution of 8 kernels = 0.010s
Measured time for sample = 0.022s
Measured tick = 110160137, total_clocks = 110160000
Test passed

*/