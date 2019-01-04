#include <iostream>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "kernels.h"

#define STB_IMAGE_IMPLEMENTATION

#include "../stb-master/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../stb-master/stb_image_write.h"

#define PYRAMID_LEVEL 6

using namespace std;

//int main(int argc, char *argv[]) {
//	int width, height, bpp;
//	float *h_data;
//	uint8_t *img_in, *img_out;
//
//	img_in = stbi_load(argv[1], &width, &height, &bpp, 3);
//
//	//stbi_write_png("/home/kca/Desktop/textureTestoriginal.png", width, height, 3, img_in, width * 3);
//
//	int usize = sizeof(uint8_t) * 3 * height * width;
//	int size = sizeof(float) * height * width;
//
//	img_out = (uint8_t *) malloc(usize/4);
//	h_data = (float *) malloc(size);
//
//	int k = 0;
//	for (int var = 0; var < 3 * height * width; var += 3) {
//		h_data[k++] = img_in[var];
//	}
//
//	// Allocate CUDA array in device memory
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
//			cudaChannelFormatKindFloat);
//	cudaArray* cuArray;
//	cudaMallocArray(&cuArray, &channelDesc, width, height);
//
//	// Copy to device memory some data located at address h_data
//	// in host memory
//	cudaMemcpyToArray(cuArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);
//
//	// Specify texture
//	struct cudaResourceDesc resDesc;
//	memset(&resDesc, 0, sizeof(resDesc));
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = cuArray;
//
//	// Specify texture object parameters
//	struct cudaTextureDesc texDesc;
//	memset(&texDesc, 0, sizeof(texDesc));
//	texDesc.addressMode[0] = cudaAddressModeClamp;
//	texDesc.addressMode[1] = cudaAddressModeClamp;
//	texDesc.filterMode = cudaFilterModeLinear;
//	texDesc.readMode = cudaReadModeElementType;
//	texDesc.normalizedCoords = 1;
//
//	// Create texture object
//	cudaTextureObject_t texObj = 0;
//	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
//
//	// Allocate result of transformation in device memory
//	float* output;
//	cudaMalloc(&output, (width) * (height) * sizeof(float) / 4);
//
//	// Invoke kernel
//	dim3 dimBlock(16, 16);
//	dim3 dimGrid((width / 2 + dimBlock.x - 1) / dimBlock.x,
//			(height / 2 + dimBlock.y - 1) / dimBlock.y);
//	transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width/2, height/2);
//
//	float *cikti = (float *) malloc(sizeof(float) * width * height / 4);
//
//	cudaMemcpy(cikti, output, size/4, cudaMemcpyDeviceToHost);
//
//	k = 0;
//	for (int var = 0; var < 3 * width * height / 4; var += 3) {
//		img_out[var] = img_out[var + 1] = img_out[var + 2] = cikti[k++];
//	}
//
//	stbi_write_png("/home/kca/Desktop/textureTest.png", width/2, height/2,
//			3, img_out, width/2 * 3);
//
//	// Destroy texture object
//	cudaDestroyTextureObject(texObj);
//
//	// Free device memory
//	cudaFreeArray(cuArray);
//	cudaFree(output);
//
//	return 0;
//}

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

	for (int i = 1; i < 2/*argc*/; ++i) {
		if (!read_Img(argv[i], images[i - 1], &width, &height, &bpp)) {
			printf("Could not read image.");
			return 0;
		}

		printf("Getting grayscale version of the image from GPU...\n");

		int nImageSize = width * height; //total pixel count of the image.
		size_t sizeOfImage = nImageSize * sizeof(uint8_t); //size of source image where each pixel is converted to uint8_t.

		cudaMalloc((void **) &d_images_rgb[i-1], sizeOfImage*3);
/*
 * Necessary for texture
 */
		size_t pitch, tex_ofs;

		cudaMallocPitch((void**)&d_images_grayscale[i-1],&pitch,width*sizeof(unsigned char),height);
/*
 * Necessary for texture
 */


//		cudaMalloc((void **) &d_images_grayscale[i-1], sizeOfImage);

		printf("width & height: %d and %d\n",width, height);

		cudaMemcpy(d_images_rgb[i-1], images[i-1], sizeOfImage*3, cudaMemcpyHostToDevice);

		int blocksize = 256;

		dim3 dimGrid, dimBlock;
		dimBlock.x = blocksize;
		dimGrid.x = nImageSize / dimBlock.x + (nImageSize%dimBlock.x == 0 ? 0 : 1);

		//grayscale conversion kernel call
		convert2_GrayScale<<< dimGrid, dimBlock >>>(d_images_grayscale[i-1], d_images_rgb[i-1], nImageSize);

		cudaDeviceSynchronize();
/*
 * TeXture Start
 */
		texRef.normalized = false;
		texRef.filterMode = cudaFilterModeLinear;

		cudaBindTexture2D (&tex_ofs, &texRef, d_images_grayscale[i-1], &texRef.channelDesc,width, height, pitch);
		cudaDeviceSynchronize();
/*
 * Texture end
 */
		int _width = (width/2);
		int _height = (height/2);

		uint8_t* output;
		cudaMalloc((void **)&output, _width * _height * sizeof(uint8_t));

		dimBlock=dim3(16, 16);
		dimGrid=dim3((_width + dimBlock.x - 1) / dimBlock.x,
					(_height + dimBlock.y - 1) / dimBlock.y);

		downsample<<<dimGrid, dimBlock>>>(output, _width, _height);

		cudaDeviceSynchronize();

		gray_images[i-1] = (uint8_t*)malloc(sizeof(uint8_t) * _width * _height);

		cudaMemcpy(gray_images[i-1], output, sizeof(uint8_t) * _width * _height, cudaMemcpyDeviceToHost);

		stbi_write_png("/home/berkay/Desktop/textureTest.png", _width, _height, 1, gray_images[i-1], _width);

		cudaUnbindTexture(texRef);
	}

	printf("Done..........\n");

	//stbi_write_png("/home/kca/Desktop/test1.png", width, height, 1, gray_images[0], width);
	//stbi_write_png("/home/kca/Desktop/test2.png", width, height, 1, gray_images[1], width);
	//stbi_write_png("/home/kca/Desktop/test3.png", width, height, 1, gray_images[2], width);
	//stbi_write_png("/home/kca/Desktop/test4.png", width, height, 1, gray_images[3], width);
	//stbi_write_png("/home/kca/Desktop/test5.png", width, height, 1, gray_images[4], width);

	return 0;
}
