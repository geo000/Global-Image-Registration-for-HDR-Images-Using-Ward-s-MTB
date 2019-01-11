#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb-master/stb_image.h"

#define BLOCK_SIZE 256
#define COLOR 256

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

//	uint8_t pixel;
	while (i < size) {

//		pixel = input[i];
		atomicAdd(&smem[input[i]], 1);

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
			return ;
		}
	}
}

__global__ void find_Mtb_Ebm(const uint8_t *input, int median, uint8_t *_mtb, uint8_t *_ebm, int _height, int _width) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = y * _width + x;

	if (input[idx] < (median - 4) || input[idx] > (median + 4)) {

		_ebm[idx] = 255;
	} else {
		_ebm[idx] = 0;
	}

	if (input[idx] < median) {

		_mtb[idx] = 0;
	} else {
		_mtb[idx] = 255;
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

__global__ void count_Errors(const uint8_t *input, int *out, int size)
{
	__shared__ int count;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
	unsigned int gridSize = BLOCK_SIZE * gridDim.x;

	count = 0;

//	uint8_t pixel;
	while (i < size) {

//		pixel = input[i];
		if(input[i] == 255)
		{
			atomicAdd(&count, 1);
		}

		i += gridSize;
	}

	if(tid == 0)
	{
		atomicAdd(out, count);
	}
}

__global__ void shift_Image(uint8_t* output, uint8_t* input, int width, int height, int x_shift, int y_shift, int j_x, int i_y, int j_width , int i_height) {

	int j = blockIdx.x * blockDim.x + threadIdx.x + j_x;
	int i = blockIdx.y * blockDim.y + threadIdx.y + i_y;

	unsigned int input_index = i * width + j;

	unsigned int output_index = y_shift * width + x_shift + i * width + j;
	//int output_index = (y_shift + i) * width + x_shift + x;

	if(i < i_height && j < j_width)
	{
		output[output_index] = input[input_index];
	}

//	if (x_shift == 0 && y_shift == 0) return;
//
//	int i_y, j_x, i_height, j_width;
//
//
//	if(y_shift < 0) { //height i
//		i_y = -y_shift;
//		i_height = height;
//	}
//	else {
//		i_y = 0;
//		i_height = height - y_shift;
//	}
//
//	if(x_shift < 0) {//width j
//		j_x = -x_shift;
//		j_width = width;
//	}
//	else {
//		j_x = 0;
//		j_width = width - x_shift;
//	}
//
//	for (int i = i_y; i < i_height; ++i) {
//		for (int j = j_x; j < j_width; ++j) {
//			output[y_shift * width + x_shift + i * width + j] = input[i * width + j];
//		}
//	}
}

bool read_Img(char *filename, uint8_t*& img, int* width, int* height, int* bpp) {

	img = stbi_load(filename, width, height, bpp, 3);

	if (img) {
		std::cout << filename << " Read Successfully\n";
		return true;
	} else {
		std::cout << filename << " Reading Failed\n";
		return false;
	}
}

#endif
