/*
 * kernels.h
 *
 *      Author: berkay
 */

//#ifndef __KERNELS_H__
//#define __KERNELS_H__

#include "device_launch_parameters.h"

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

__global__ void AND(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	index = y * width + x;
	output[index] = left[index] & right[index];
}

__global__ void XOR(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	index = y * width + x;
	output[index] = left[index] ^ right[index];
}
//
//#endif


