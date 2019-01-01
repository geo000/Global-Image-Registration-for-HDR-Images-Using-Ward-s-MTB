//
// Created by berkay on 31.12.2018.
//

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_mtgp32_kernel.h>

#define STB_IMAGE_IMPLEMENTATION

#include "../stb-master/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../stb-master/stb_image_write.h"

using namespace std;

__global__ void convert2_GrayScale(uint8_t* gray, uint8_t *img, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		int index = idx * 3;
		gray[idx] = (54 * img[index] + 183 * img[index + 1]
				+ 19 * img[index + 2]) / 256.0f;
	}
}

bool read_Img(char *filename, uint8_t* img, int* width, int* height, int* bpp) {

	img = stbi_load(filename, width, height, bpp, 3);

	if (img) {
		std::cout << filename << " Read Successfully\n";
		return true;
	} else {
		std::cout << filename << " Reading Failed\n";
		return false;
	}
}

int main(int argc, char* argv[]) {
	uint8_t* images[argc - 1]; //that many images are given as input.
	uint8_t* gray_images[argc - 1];
	int width, height, bpp;
	int img_count = argc - 1;
	uint8_t* d_images_rgb[img_count];
	uint8_t* d_images_grayscale[img_count];

	if (argc == 1) {
		printf("Please supply the input images.");
		return 0;
	}

	for (int i = 1; i < 2/*TODO argc*/; ++i) {
		if (!read_Img(argv[i], images[i - 1], &width, &height, &bpp)) {
			printf("Could not read image.");
			return 0;
		}

		printf("Getting grayscale version of the image from GPU...\n");

		int nImageSize = width * height; //total pixel count of the image.
		size_t sizeOfImage = nImageSize * sizeof(uint8_t); //size of source image where each pixel is converted to uint8_t.

		cudaMalloc((void **) &d_images_rgb[i-1], sizeOfImage*3);
		cudaMalloc((void **) &d_images_grayscale[i-1], sizeOfImage);

		printf("width & height: %d and %d\n",width, height);

		//stbi_write_png("/home/kca/Desktop/test2.png", width, height, 3, images[i-1], width*3);


		cudaMemcpy(d_images_rgb[i-1], images[i-1], sizeOfImage, cudaMemcpyHostToDevice);

		int blocksize = 256;

		dim3 dimGrid, dimBlock;
		dimBlock.x = blocksize;
		dimGrid.x = nImageSize / dimBlock.x + (nImageSize%dimBlock.x == 0 ? 0 : 1);

		//grayscale conversion kernel call

		convert2_GrayScale<<< dimGrid, dimBlock >>>(d_images_grayscale[i-1], d_images_rgb[i-1], nImageSize);

		cudaDeviceSynchronize();

		gray_images[i-1] = (uint8_t*)malloc(sizeof(uint8_t) * sizeOfImage);

		cudaMemcpy(gray_images[i-1], d_images_grayscale[i-1], sizeOfImage, cudaMemcpyDeviceToHost);

		stbi_write_png("/home/kca/Desktop/test.png", width, height, 1, gray_images[i-1], width);

		printf("Done..........");
	}

	return 0;
}










