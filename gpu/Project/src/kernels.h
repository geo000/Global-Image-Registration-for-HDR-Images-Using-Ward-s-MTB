/*
 * kernels.h
 *
 *      Author: berkay
 */

//#ifndef __KERNELS_H__
//#define __KERNELS_H__

#include "device_launch_parameters.h"
#define BLOCK_SIZE 256
#define COLOR 256

//__global__ void transformKernel(float* output, cudaTextureObject_t texObj,
//		int width, int height) {
//
//	// Calculate normalized texture coordinates
//	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float u = x / (float) width ;
//	float v = y / (float) height ;
//
//	// Read from texture and write to global memory
//	output[y * width + x] = tex2D<float>(texObj, u, v);
//
//}

texture<unsigned char,  2,  cudaReadModeNormalizedFloat> texRef;

__global__ void downsample(uint8_t* output, int width, int height) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Read from texture and write to global memory
	output[y * width + x] =(tex2D<unsigned char>(texRef, 2*x+1, 2*y+1))*255;
}

__global__ void convert2_GrayScale(uint8_t* gray, uint8_t *img, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		int index = idx * 3;
		gray[idx] = (54 * img[index] + 183 * img[index + 1]
				+ 19 * img[index + 2]) / 256.0f;
	}
}

__global__ void histogram_smem_atomics(const uint8_t *input, int *out, int size)
{
	__shared__ int smem[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
	unsigned int gridSize = BLOCK_SIZE * gridDim.x;

	smem[tid] = 0;
	__syncthreads();

	uint8_t pixel;
	while (i < size) {

		pixel = input[i];
		atomicAdd(&smem[pixel], 1);

		i += gridSize;
	}
	__syncthreads();

	out[blockIdx.x * BLOCK_SIZE + tid] = smem[tid];
}

__global__ void histogram_final_accum(int n, int *out)
{
  int tid = threadIdx.x;
  int i = tid;
  int total = 0;

  while(i < n)
  {
	  total += out[i];
	  i += BLOCK_SIZE;
  }
  __syncthreads();

  out[tid] = total;

}

__global__ void find_Median(int n, int *hist, int* median)
{
	int half_way = n / 2;

	int sum = 0;

	for (int k = 0; k < COLOR; k++) {
		sum += hist[k];
		if (sum > half_way) {
			*median = k;
			return;
		}
	}
}


__global__ void AND(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	output[index] = left[index] & right[index];
}

__global__ void XOR(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	output[index] = left[index] ^ right[index];
}


//
//#endif


