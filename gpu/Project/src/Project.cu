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

	if(idx < size)
	{
		int index = idx*3;
		gray[idx] = (54 * img[index] + 183 * img[index + 1] + 19 * img[index + 2]) / 256.0f;
	}
}

bool read_Img(char *filename, uint8_t* img, int* width, int* height, int* bpp)
{

    img = stbi_load(filename, width, height, bpp, 3);

    if (img) {
        std::cout << filename << " Read Successfully\n";
        return true;
    } else {
        std::cout << filename << " Reading Failed\n";
        return false;
    }
}

int main(int argc, char* argv[])
{
	uint8_t* images[argc-1];
	int width,height,bpp;

	for (int i = 1; i < argc; ++i) {

		if(!read_Img(argv[i],images[i-1], &width, &height, &bpp))
		{
			return 0;
		}
	}

	return 0;
}
