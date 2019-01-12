#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb-master/stb_image.h"

#define BLOCK_SIZE 256
#define COLOR 256
#define PYRAMID_LEVEL 6

typedef struct shift_pair {
    shift_pair(int _x, int _y) {
        x = _x;
        y = _y;
    }

    int x;
    int y;
} shift_pair;

// Simple transformation kernel
__global__ void transformKernel(float* output, cudaTextureObject_t texObj,
		int width, int height) {

	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float) width ;
	float v = y / (float) height ;

	// Read from texture and write to global memory

	output[y * width + x] = tex2D<float>(texObj, u, v);

}


//__global__ void downsample(uint8_t* output, int width, int height) {
//
//	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	// Read from texture and write to global memory
//	//if(x< width && y< height)
//	//{
//	output[y * width + x] =(tex2D<unsigned char>(texRef, 2*x+1, 2*y+1))*255;//}
//}

__global__ void convert2_GrayScale(float* gray, uint8_t *img, int size, int width) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * width + x;

	if (idx < size) {
		int index = idx * 3;
		gray[idx] = (54 * img[index] + 183 * img[index + 1]
				+ 19 * img[index + 2]) / 256.0f;
	}
}

__global__ void histogram_smem_atomics(const float *input, int *out, int size)
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
		atomicAdd(&smem[(int)input[i]], 1);

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

__global__ void find_Mtb_Ebm(const float *input, int* median, uint8_t *_mtb, uint8_t *_ebm, int _height, int _width) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = y * _width + x;

	if (input[idx] < (*median - 4) || input[idx] > (*median + 4)) {

		_ebm[idx] = 255;
	} else {
		_ebm[idx] = 0;
	}

	if (input[idx] < *median) {

		_mtb[idx] = 0;
	} else {
		_mtb[idx] = 255;
	}
}

__global__ void AND(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	if (index <size) {
		output[index] = left[index] & right[index];
	}
}

__global__ void XOR(uint8_t* output, uint8_t* left, uint8_t *right, int width, int size) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	if (index <size) {
		output[index] = left[index] ^ right[index];
	}
}

__global__ void count_Errors(const uint8_t *input, int *out, int size)
{
	__shared__ int count;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
	unsigned int gridSize = BLOCK_SIZE * gridDim.x;

	count = 0;

	while (i < size) {

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


__global__ void finalShift(uint8_t* output, float* input, int width, int height, int x_shift, int y_shift, int j_x, int i_y, int j_width , int i_height) {

	int j = blockIdx.x * blockDim.x + threadIdx.x + j_x;
	int i = blockIdx.y * blockDim.y + threadIdx.y + i_y;

	unsigned int input_index = i * width + j;

	unsigned int output_index = y_shift * width + x_shift + i * width + j;

	if(i < i_height && j < j_width)
	{
		output[output_index] = input[input_index];
	}
}

//__global__ void calculateOffsetOfTwoImages(int first_index, int second_index, uint8_t** mtb, uint8_t** ebm, uint8_t** shifted_mtb, uint8_t** shifted_ebm, int width, int height, int* res)
//{
//	dim3 dimGrid, dimBlock;
//	int tmpWidth = width/(pow(2,PYRAMID_LEVEL-1));
//	int tmpHeight = height/(pow(2,PYRAMID_LEVEL-1));
//	int tmpNImageSize= tmpWidth * tmpHeight;
//
//	int curr_level = PYRAMID_LEVEL - 1;
//	int curr_offset_x = 0;
//	int curr_offset_y = 0;
//	int offset_return_x = 0;
//	int offset_return_y = 0;
//
//	for (int k = curr_level; k >= 0; --k, tmpWidth *= 2, tmpHeight *= 2 , tmpNImageSize *= 4) {
//		curr_offset_x = 2 * offset_return_x;
//		curr_offset_y = 2 * offset_return_y;
//
//		int min_error = 255 * height * width;
//
//		for (int i = -1; i <= 1; ++i) {
//			for (int j = -1; j <= 1; ++j) {
//				int xs = curr_offset_x + i;
//				int ys = curr_offset_y + j;
//
//
//				int x_shift=xs, y_shift=ys; //TODO check those
//
//				int j_x, i_y, j_width, i_height;
//
//				if(y_shift < 0) { //height i
//					i_y = -y_shift;
//					i_height = tmpHeight;
//				}
//				else {
//					i_y = 0;
//					i_height = tmpHeight - y_shift;
//				}
//
//				if(x_shift < 0) {//width j
//					j_x = -x_shift;
//					j_width = tmpWidth;
//				}
//				else {
//					j_x = 0;
//					j_width = tmpWidth - x_shift;
//				}
//
//				dimBlock=dim3(16, 16);
//				dimGrid=dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
//							((i_height) + dimBlock.y - 1) / dimBlock.y);
//
//				cudaMemset((void **)shifted_mtb[second_index * PYRAMID_LEVEL + k], 0, tmpNImageSize * sizeof(uint8_t));
//				cudaMemset((void **)shifted_ebm[second_index * PYRAMID_LEVEL + k], 0, tmpNImageSize * sizeof(uint8_t));
//
//				shift_Image<<<dimGrid, dimBlock>>>(shifted_mtb[second_index * PYRAMID_LEVEL + k], mtb[second_index * PYRAMID_LEVEL + k], tmpWidth, tmpHeight, xs, ys, j_x, i_y, j_width , i_height);
//				shift_Image<<<dimGrid, dimBlock>>>(shifted_ebm[second_index * PYRAMID_LEVEL + k], ebm[second_index * PYRAMID_LEVEL + k], tmpWidth, tmpHeight, xs, ys, j_x, i_y, j_width , i_height);
//
//				uint8_t *xor_result;
//				cudaMalloc((void **)&xor_result, tmpNImageSize * sizeof(uint8_t));
//
//				dimBlock=dim3(16, 16);
//				dimGrid=dim3(((tmpWidth) + dimBlock.x - 1) / dimBlock.x,
//							((tmpHeight) + dimBlock.y - 1) / dimBlock.y);
//
//				XOR<<<dimGrid, dimBlock>>>(xor_result, mtb[first_index * PYRAMID_LEVEL + k], shifted_mtb[second_index * PYRAMID_LEVEL + k], tmpWidth, tmpNImageSize);
//
//				uint8_t *after_first_and;
//				cudaMalloc((void **)&after_first_and, tmpNImageSize * sizeof(uint8_t));
//
//				uint8_t *after_second_and;
//				cudaMalloc((void **)&after_second_and, tmpNImageSize * sizeof(uint8_t));
//
//				AND<<<dimGrid, dimBlock>>>(after_first_and,ebm[first_index * PYRAMID_LEVEL + k], xor_result, tmpWidth, tmpNImageSize);
//
//				AND<<<dimGrid, dimBlock>>>(after_second_and,shifted_ebm[second_index * PYRAMID_LEVEL + k], after_first_and, tmpWidth, tmpNImageSize);
//
//				int* err;
//				int error;
//				//cudaMalloc((void **)&err, sizeof(int));
//
//				count_Errors<<<32, 256>>>(after_second_and, err, tmpNImageSize);
//
//				//cudaMemcpy(&error, err, sizeof(int), cudaMemcpyDeviceToHost);
//				error=*err;
//				if (error < min_error) {
//					offset_return_x = xs;
//					offset_return_y = ys;
//					min_error = error;
//				}
//				//cudaFree(err);
//			}
//		}
//	}
//	res[0] = curr_offset_x;
//	res[1] = curr_offset_y;
//
//}


bool read_Img(char *filename, uint8_t*& img, int* width, int* height, int* bpp) {

	img = stbi_load(filename, width, height, bpp, 3);

	if (img) {
	//	std::cout << filename << " Read Successfully\n";
		return true;
	} else {
	//	std::cout << filename << " Reading Failed\n";
		return false;
	}
}

#endif
