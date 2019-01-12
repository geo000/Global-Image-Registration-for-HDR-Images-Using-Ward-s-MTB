#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "kernels.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb-master/stb_image_write.h"

using namespace std;

int main(int argc, char* argv[]) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (argc == 1) {
		printf("Please supply the input images.");
		return 0;
	}

	int img_count = argc - 1;
	int width, height, bpp;
	uint8_t* rgb_images[img_count]; //that many images are given as input.
	uint8_t* shifted_rgb_images[img_count];
	uint8_t* d_rgb_images[img_count];
	float* gray_image[PYRAMID_LEVEL * img_count];
	uint8_t* mtb[PYRAMID_LEVEL * img_count];
	uint8_t* ebm[PYRAMID_LEVEL * img_count];
	uint8_t* shifted_mtb[PYRAMID_LEVEL * img_count];
	uint8_t* shifted_ebm[PYRAMID_LEVEL * img_count];


	//cudaMalloc((void **) &d_images, sizeof(GPUImage)*PYRAMID_LEVEL*img_count);

	for (int i = 1; i <= img_count; ++i) {
		if (!read_Img(argv[i], rgb_images[i - 1], &width, &height, &bpp)) {
			printf("Could not read image.");
			return 0;
		}

		int nImageSize = width * height; //total pixel count of the image.
		size_t sizeOfImage = nImageSize * sizeof(uint8_t); //size of source image where each pixel is converted to uint8_t.

		cudaMalloc((void **) &d_rgb_images[i-1], sizeOfImage*3);
		cudaMemcpy(d_rgb_images[i-1], rgb_images[i-1], sizeOfImage*3, cudaMemcpyHostToDevice);

		int tmpSizeOfImage = sizeOfImage;
		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = nImageSize;

//		texRef.normalized = false;
//		texRef.filterMode = cudaFilterModeLinear;
//		size_t pitch[30], tex_ofs[30];
		for(int j = 0; j < PYRAMID_LEVEL; j++, tmpSizeOfImage/=4, tmpWidth/=2, tmpHeight/=2, tmpNImageSize/=4){

//			texRef.normalized = false;
//			texRef.filterMode = cudaFilterModeLinear;

			//cudaMallocPitch((void**)&(gray_image[(i-1) * PYRAMID_LEVEL + j]), &(pitch[(i-1) * PYRAMID_LEVEL + j]), tmpWidth, tmpHeight);

			// Allocate CUDA array in device memory
//			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
//						cudaChannelFormatKindFloat);
//			cudaMallocArray(&(gray_image[(i-1) * PYRAMID_LEVEL + j]), &channelDesc, tmpWidth, tmpHeight);
			cudaMalloc((void **) &(gray_image[(i-1) * PYRAMID_LEVEL + j]), tmpNImageSize*sizeof(float));
			cudaMalloc((void **) &(mtb[(i-1) * PYRAMID_LEVEL + j]), tmpSizeOfImage);
			cudaMalloc((void **) &(ebm[(i-1) * PYRAMID_LEVEL + j]), tmpSizeOfImage);
			cudaMalloc((void **) &(shifted_mtb[(i-1) * PYRAMID_LEVEL + j]), tmpSizeOfImage);
			cudaMalloc((void **) &(shifted_ebm[(i-1) * PYRAMID_LEVEL + j]), tmpSizeOfImage);

			dim3 dimGrid, dimBlock;
			dimBlock=dim3(16, 16);
			dimGrid=dim3((tmpWidth + dimBlock.x - 1) / dimBlock.x,
						(tmpHeight + dimBlock.y - 1) / dimBlock.y);

			if(j==0){

				convert2_GrayScale<<< dimGrid, dimBlock >>>(gray_image[(i-1) * PYRAMID_LEVEL + j], d_rgb_images[i-1], tmpNImageSize, tmpWidth);
			} else {
//				cudaBindTexture2D (&tex_ofs, &texRef, gray_image[(i-1) * PYRAMID_LEVEL + j-1], &texRef.channelDesc, tmpWidth*2, tmpHeight*2, pitch);
//				texRef = ref[(i-1) * PYRAMID_LEVEL + j];

				cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
						cudaChannelFormatKindFloat);
				cudaArray* cuArray;
				cudaMallocArray(&cuArray, &channelDesc, tmpWidth*2, tmpHeight*2);

				// Copy to device memory some data located at address h_data
				// in host memory
				cudaMemcpyToArray(cuArray, 0, 0, gray_image[(i-1) * PYRAMID_LEVEL + j -1], tmpNImageSize*sizeof(float)*4, cudaMemcpyDeviceToDevice);


				// Specify texture
				struct cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = cuArray;

				// Specify texture object parameters
				struct cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = cudaAddressModeClamp;
				texDesc.addressMode[1] = cudaAddressModeClamp;
				texDesc.filterMode = cudaFilterModeLinear;
				texDesc.readMode = cudaReadModeElementType;
				texDesc.normalizedCoords = 1;

				// Create texture object
				cudaTextureObject_t texObj = 0;
				cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

				// Invoke kernel
				dim3 dimBlock(16, 16);
				dim3 dimGrid((tmpWidth + dimBlock.x - 1) / dimBlock.x,
						(tmpHeight + dimBlock.y - 1) / dimBlock.y);

				transformKernel<<<dimGrid, dimBlock>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], texObj, tmpWidth, tmpHeight);

				cudaDestroyTextureObject(texObj);

				// Free device memory
				cudaFreeArray(cuArray);
//				downsample<<<dimGrid, dimBlock>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], tmpWidth, tmpHeight);
//				cudaDeviceSynchronize();
//				cudaUnbindTexture(texRef);gray_image
//				cudaFree(gray_image[(i-1) * PYRAMID_LEVEL + j-1]);
			}

//TODO dynamic parallelism kullanamadik kernel icinde kernel. cok kasti. cudaMalloc vs dinamik size'da yapamadik kernel icinde.
//TODO texture cok vaktimizi aldi. (2 gün totalde) globalden texRef alan yontem imaj piramidinin bazi levellarinda calismadi,
//TODO benchmark using 2 kernels of histogram finding, and 1 merged kernel.
//TODO findMedian kernel'ini de gom ustteki merge haline.

//			char str[12];
//			sprintf(str, "%d.png", (i-1) * PYRAMID_LEVEL + j);
//			char path[80]="/home/kca/Desktop/test_mtb";
//			strcat(path, str);
//
//			float* tmpmtb = (float *)malloc(sizeof(float)*tmpNImageSize);
//			cudaMemcpy(tmpmtb, gray_image[(i-1) * PYRAMID_LEVEL + j], sizeof(float)*tmpNImageSize, cudaMemcpyDeviceToHost);
//			uint8_t* tmpmtbuint;
//			tmpmtbuint = (uint8_t*)malloc(tmpNImageSize);
//			for (int var = 0; var < tmpNImageSize; ++var) {
//				tmpmtbuint[var] = tmpmtb[var];
//			}
//			stbi_write_png(path, tmpWidth, tmpHeight, 1, tmpmtbuint, tmpWidth);
//
			dimBlock=dim3(BLOCK_SIZE);
			dimGrid=dim3(32);

			int* hist;
			cudaMalloc((void **)&hist, BLOCK_SIZE * sizeof(int) * 32);

			histogram_smem_atomics<<<dimGrid, dimBlock>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], hist, tmpNImageSize);
			histogram_final_accum<<<1, 256>>>(BLOCK_SIZE*dimGrid.x, hist);

			int* median; cudaMalloc((void **)&median, sizeof(int));
			find_Median<<<1, 1>>>(tmpNImageSize, hist, median);

			dimBlock=dim3(16, 16);
			dimGrid=dim3((tmpWidth + dimBlock.x - 1) / dimBlock.x,
						(tmpHeight + dimBlock.y - 1) / dimBlock.y);

			find_Mtb_Ebm<<<dimGrid, dimBlock>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], median, mtb[(i-1) * PYRAMID_LEVEL + j],
					ebm[(i-1) * PYRAMID_LEVEL + j], tmpHeight, tmpWidth);

//			uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize);
//			cudaMemcpy(tmpmtb, mtb[(i-1) * PYRAMID_LEVEL + j], sizeof(uint8_t)*tmpNImageSize, cudaMemcpyDeviceToHost);
//			stbi_write_png("/home/kca/Desktop/test_mtb.png", tmpWidth, tmpHeight, 1, tmpmtb, tmpWidth);
//			int c = 0;

//			int h_median;
//			cudaMemcpy(&h_median, median, sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(hist);
			cudaFree(median);

//			cudaBindTexture2D (&(tex_ofs[(i-1) * PYRAMID_LEVEL + j]), &texRef, gray_image[(i-1) * PYRAMID_LEVEL + j], &texRef.channelDesc,
//					tmpWidth, tmpHeight, pitch[(i-1) * PYRAMID_LEVEL + j]);


		}

//		uint8_t* shifted;
//		cudaMalloc((void **)&shifted, width * height *  sizeof(uint8_t));
//
//	    cudaMemset(shifted, 255, width * height *  sizeof(uint8_t));
//
//	    int x_shift=-200, y_shift=200;
//
//	    int j_x, i_y, j_width, i_height;
//
//	    if(y_shift < 0) { //height i
//			i_y = -y_shift;
//			i_height = height;
//		}
//		else {
//			i_y = 0;
//			i_height = height - y_shift;
//		}
//
//		if(x_shift < 0) {//width j
//			j_x = -x_shift;
//			j_width = width;
//		}
//		else {
//			j_x = 0;
//			j_width = width - x_shift;
//		}
//
//
//
//
//		dimBlock=dim3(16, 16);
//		dimGrid=dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
//					((i_height) + dimBlock.y - 1) / dimBlock.y);
//
//		shift_Image<<<dimGrid, dimBlock>>>(shifted, d_images_grayscale[i-1], width, height, x_shift, y_shift, j_x, i_y, j_width , i_height);
	}



//**********************************************************************************************************************************
    std::vector<shift_pair> all_shifts;

    int mid_img_index = img_count / 2 + 1;

    for (int m = mid_img_index - 1; m >= 0; --m) {
        all_shifts.emplace_back(calculateOffset(m+1, m, width, height,gray_image, mtb, ebm, shifted_mtb, shifted_ebm));
    }

    //cout << " ilk parttaki imajlari shiftliyoruz tek tek ..." << endl;

    int k = 0, eskiTotalX = 0, eskiTotalY = 0;
    for (int m = mid_img_index - 1; m >= 0; --m) {

//    	all_images[m].finalShift(all_shifts[k].x + eskiTotalX, all_shifts[k].y + eskiTotalY);

		char str[12];
		sprintf(str, "m%d-k%d.png", m,k);
		char path[80]="/home/berkay/Desktop/out/shiftedGray";
		strcat(path, str);



		int x_shift=all_shifts[k].x + eskiTotalX, y_shift=all_shifts[k].y + eskiTotalY; //TODO check those

		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = tmpWidth * tmpHeight;

		cudaMalloc((void**)&shifted_rgb_images[m], 3 * tmpNImageSize);
		cudaMemset(shifted_rgb_images[m], 0, 3*tmpNImageSize * sizeof(uint8_t));

		int j_x, i_y, j_width, i_height;

		if(y_shift < 0) { //height i
			i_y = -y_shift;
			i_height = tmpHeight;
		}
		else {
			i_y = 0;
			i_height = tmpHeight - y_shift;
		}

		if(x_shift < 0) {//width j
			j_x = -x_shift;
			j_width = tmpWidth;
		}
		else {
			j_x = 0;
			j_width = tmpWidth - x_shift;
		}

		dim3 dimBlock=dim3(16, 16);
		dim3 dimGrid=dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
					((i_height) + dimBlock.y - 1) / dimBlock.y);

//		uint8_t* tmpShiftedGray;
//		cudaMalloc((void **) &tmpShiftedGray,tmpNImageSize);
//		cudaMemset(tmpShiftedGray, 0, tmpNImageSize * sizeof(uint8_t));

		RGB_shift_Image<<<dimGrid, dimBlock>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);

		uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize*3);
		cudaMemcpy(tmpmtb, shifted_rgb_images[m], sizeof(uint8_t)*tmpNImageSize*3, cudaMemcpyDeviceToHost);

		stbi_write_png(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth*3);

        eskiTotalX += all_shifts[k].x;
        eskiTotalY += all_shifts[k].y;
        k++;
        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }

    //cout << "ikinci part baslar ..." << endl;
    all_shifts.clear();

    for (int m = mid_img_index + 1; m < img_count; ++m) {
        all_shifts.emplace_back(calculateOffset(m-1, m, width, height, gray_image,mtb, ebm, shifted_mtb, shifted_ebm));
    }

    //cout << " ikinci parttaki imajlari shiftliyoruz tek tek ..." << endl;

    k = 0;
    eskiTotalX = 0;
    eskiTotalY = 0;
    for (int m = mid_img_index + 1; m < img_count; ++m) {
        //all_images[m].finalShift(all_shifts[k].x + eskiTotalX, all_shifts[k].y + eskiTotalY);



    	char str[12];
		sprintf(str, "m%d-k%d.png", m,k);
		char path[80]="/home/berkay/Desktop/out/shiftedGray";
		strcat(path, str);



		int x_shift=all_shifts[k].x + eskiTotalX, y_shift=all_shifts[k].y + eskiTotalY; //TODO check those

		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = tmpWidth * tmpHeight;

		cudaMalloc((void**)&shifted_rgb_images[m], 3 * tmpNImageSize);
		cudaMemset(shifted_rgb_images[m], 0, 3*tmpNImageSize * sizeof(uint8_t));

		int j_x, i_y, j_width, i_height;

		if(y_shift < 0) { //height i
			i_y = -y_shift;
			i_height = tmpHeight;
		}
		else {
			i_y = 0;
			i_height = tmpHeight - y_shift;
		}

		if(x_shift < 0) {//width j
			j_x = -x_shift;
			j_width = tmpWidth;
		}
		else {
			j_x = 0;
			j_width = tmpWidth - x_shift;
		}

		dim3 dimBlock=dim3(16, 16);
		dim3 dimGrid=dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
					((i_height) + dimBlock.y - 1) / dimBlock.y);

//		uint8_t* tmpShiftedGray;
//		cudaMalloc((void **) &tmpShiftedGray,tmpNImageSize);
//		cudaMemset(tmpShiftedGray, 0, tmpNImageSize * sizeof(uint8_t));

		RGB_shift_Image<<<dimGrid, dimBlock>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);

		uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize*3);
		cudaMemcpy(tmpmtb, shifted_rgb_images[m], sizeof(uint8_t)*tmpNImageSize*3, cudaMemcpyDeviceToHost);

		stbi_write_png(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth*3);



        eskiTotalX += all_shifts[k].x;
        eskiTotalY += all_shifts[k].y;
        k++;
        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }



    //print original grayscale img

	char str[12];
	sprintf(str, ".png");
	char path[80]="/home/berkay/Desktop/out/orgi";
	strcat(path, str);

	stbi_write_png(path, width, height, 3, rgb_images[mid_img_index], width*3);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<milliseconds<<endl;




	//printf("Done..........\n");

	//stbi_write_png("/home/kca/Desktop/test1.png", width, height, 1, gray_images[0], width);
	//stbi_write_png("/home/kca/Desktop/test2.png", width, height, 1, gray_images[1], width);
	//stbi_write_png("/home/kca/Desktop/test3.png", width, height, 1, gray_images[2], width);
	//stbi_write_png("/home/kca/Desktop/test4.png", width, height, 1, gray_images[3], width);
	//stbi_write_png("/home/kca/Desktop/test5.png", width, height, 1, gray_images[4], width);

	return 0;
}
