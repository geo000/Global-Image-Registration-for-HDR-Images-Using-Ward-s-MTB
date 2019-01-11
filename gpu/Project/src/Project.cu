#include <iostream>
#include "cuda_runtime.h"
#include "kernels.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb-master/stb_image_write.h"

#define PYRAMID_LEVEL 6

using namespace std;

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

		dimBlock=dim3(BLOCK_SIZE);

		dimGrid=dim3(32);

		int* out;
		cudaMalloc((void **)&out, BLOCK_SIZE * sizeof(int) * 32);

		histogram_smem_atomics<<<dimGrid, dimBlock>>>(d_images_grayscale[i-1], out, width*height);

		cudaDeviceSynchronize();

		histogram_final_accum<<<1, 256>>>(BLOCK_SIZE*dimGrid.x, out);

//		int res[256];

//		cudaMemcpy(res, out, sizeof(int) * 256, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		int* median;
		cudaMalloc((void **)&median, sizeof(int));

		find_Median<<<1, 1>>>(width*height, out, median);



		uint8_t* shifted;
		cudaMalloc((void **)&shifted, width * height *  sizeof(uint8_t));

	    cudaMemset(shifted, 255, width * height *  sizeof(uint8_t));

	    int x_shift=-200, y_shift=200;

	    int j_x, i_y, j_width, i_height;

	    if(y_shift < 0) { //height i
			i_y = -y_shift;
			i_height = height;
		}
		else {
			i_y = 0;
			i_height = height - y_shift;
		}

		if(x_shift < 0) {//width j
			j_x = -x_shift;
			j_width = width;
		}
		else {
			j_x = 0;
			j_width = width - x_shift;
		}




		dimBlock=dim3(16, 16);
		dimGrid=dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
					((i_height) + dimBlock.y - 1) / dimBlock.y);





		shift_Image<<<dimGrid, dimBlock>>>(shifted, d_images_grayscale[i-1], width, height, x_shift, y_shift, j_x, i_y, j_width , i_height);







//		int med[1];
//		cudaMemcpy(med, median, sizeof(int), cudaMemcpyDeviceToHost);

		gray_images[i-1] = (uint8_t*)malloc(sizeof(uint8_t) * width * height);

		cudaMemcpy(gray_images[i-1], shifted, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);

		stbi_write_png("/home/berkay/Desktop/textureTest.png", width, height, 1, gray_images[i-1], width);
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
