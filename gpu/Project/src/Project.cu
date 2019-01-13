#include <iostream>
#include <vector>
#include <pthread.h>
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

	int num_threads = img_count;

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

	cudaStream_t streams[img_count];

	for (int i = 1; i <= img_count; ++i) {
		if (!read_Img(argv[i], rgb_images[i - 1], &width, &height, &bpp)) {
			printf("Could not read image.");
			return 0;
		}

		int nImageSize = width * height; //total pixel count of the image.
		size_t sizeOfImage = nImageSize * sizeof(uint8_t); //size of source image where each pixel is converted to uint8_t.

		cudaStreamCreate(&(streams[i-1]));
		cudaHostRegister(rgb_images[i - 1], sizeOfImage*3, 0);

		cudaMalloc((void **) &d_rgb_images[i-1], sizeOfImage*3);
		cudaMemcpyAsync(d_rgb_images[i-1], rgb_images[i-1], sizeOfImage*3, cudaMemcpyHostToDevice, streams[i-1]);

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
				convert2_GrayScale<<< dimGrid, dimBlock, 0, streams[i-1] >>>(gray_image[(i-1) * PYRAMID_LEVEL + j], d_rgb_images[i-1], tmpNImageSize, tmpWidth);
				//cudaDeviceSynchronize();
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

				transformKernel<<<dimGrid, dimBlock, 0, streams[i-1]>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], texObj, tmpWidth, tmpHeight);

				//cudaDestroyTextureObject(texObj);

				// Free device memory
				//cudaFreeArray(cuArray);
//				downsample<<<dimGrid, dimBlock>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], tmpWidth, tmpHeight);
//				cudaDeviceSynchronize();
//				cudaUnbindTexture(texRef);gray_image
//				cudaFree(gray_image[(i-1) * PYRAMID_LEVEL + j-1]);
			}

//TODO dynamic parallelism kullanamadik kernel icinde kernel. cok kasti. cudaMalloc vs dinamik size'da yapamadik kernel icinde.
//TODO texture cok vaktimizi aldi. (2 gün totalde) globalden texRef alan yontem imaj piramidinin bazi levellarinda calismadi,
//TODO benchmark using 2 kernels of histogram finding, and 1 merged kernel.
//TODO findMedian kernel'ini de gom ustteki merge haline.
//TODO write'ler olmadan, readler dahil 1080 -> 593.515ms. gtx850m -> 1638.57 ms.
//TODO preprocessing kisminda img_count tane stream actik, her mtb ebm find kernel'i kendi streaminde. memcpyAsync. gtx850m->1495.27 ms oldu.
//TODO asil algoritmanin kostugu kismi da her imajin kendi stream'ine koyduk. ama calculateOffsetError if checki sequential hala. gtx850m -> 674.244ms
//TODO simdi kernel merge, stream vs yapacagiz bakalim hizlanacak mi...

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

			histogram_smem_atomics<<<dimGrid, dimBlock, 0, streams[i-1]>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], hist, tmpNImageSize);
			histogram_final_accum<<<1, 256, 0, streams[i-1]>>>(BLOCK_SIZE*dimGrid.x, hist);

			int* median; cudaMalloc((void **)&median, sizeof(int));
			find_Median<<<1, 1, 0, streams[i-1]>>>(tmpNImageSize, hist, median);

			dimBlock=dim3(16, 16);
			dimGrid=dim3((tmpWidth + dimBlock.x - 1) / dimBlock.x,
						(tmpHeight + dimBlock.y - 1) / dimBlock.y);

			find_Mtb_Ebm<<<dimGrid, dimBlock, 0 , streams[i-1]>>>(gray_image[(i-1) * PYRAMID_LEVEL + j], median, mtb[(i-1) * PYRAMID_LEVEL + j],
					ebm[(i-1) * PYRAMID_LEVEL + j], tmpHeight, tmpWidth);

//			uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize);
//			cudaMemcpy(tmpmtb, mtb[(i-1) * PYRAMID_LEVEL + j], sizeof(uint8_t)*tmpNImageSize, cudaMemcpyDeviceToHost);
//			stbi_write_png("/home/kca/Desktop/test_mtb.png", tmpWidth, tmpHeight, 1, tmpmtb, tmpWidth);
//			int c = 0;

//			int h_median;
//			cudaMemcpy(&h_median, median, sizeof(int), cudaMemcpyDeviceToHost);

			//cudaFree(hist);
			//cudaFree(median);

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


	cudaDeviceSynchronize(); //wait for all streams to finish up.

	for (int var = 0; var < img_count; ++var) {
		cudaHostUnregister(rgb_images[var]);
	}

//**********************************************************************************************************************************
    shift_pair all_shifts[img_count];

    int mid_img_index = img_count / 2 ; //TODO belki +1.

    for (int m = mid_img_index - 1; m >= 0; --m) {

//    	cudaEvent_t startstream, stopstream;
//    	cudaEventCreate(&startstream);
//    	cudaEventCreate(&stopstream);
//    	cudaEventRecord(start, streams[m]);

    	calculateOffset(&(all_shifts[m]), (streams[m]), m+1, m, width, height,gray_image, mtb, ebm, shifted_mtb, shifted_ebm);
//        all_shifts[m]=(calculateOffset(m+1, m, width, height,gray_image, mtb, ebm, shifted_mtb, shifted_ebm));

//    	cudaEventRecord(stopstream, streams[m]);
//    	cudaEventSynchronize(stopstream);
//    	float milliseconds = 0;
//    	cudaEventElapsedTime(&milliseconds, startstream, stopstream);
//    	cout<<"finalshift "<<m<<" : "<<milliseconds<<endl;
    }

    for (int m = mid_img_index + 1; m < img_count; ++m) {

//    	cudaEvent_t startstream, stopstream;
//    	cudaEventCreate(&startstream);
//    	cudaEventCreate(&stopstream);
//    	cudaEventRecord(start, streams[m]);

    	calculateOffset(&(all_shifts[m]), (streams[m]), m-1, m, width, height, gray_image,mtb, ebm, shifted_mtb, shifted_ebm);
    //	all_shifts[m]=(calculateOffset(m-1, m, width, height, gray_image,mtb, ebm, shifted_mtb, shifted_ebm));
//
//    	cudaEventRecord(stopstream, streams[m]);
//    	cudaEventSynchronize(stopstream);
//    	float milliseconds = 0;
//    	cudaEventElapsedTime(&milliseconds, startstream, stopstream);
//    	cout<<"finalshift "<<m<<" : "<<milliseconds<<endl;


    }

    cudaDeviceSynchronize();

    //cout << " ilk parttaki imajlari shiftliyoruz tek tek ..." << endl;

	int eskiTotalX = 0, eskiTotalY = 0;
    for (int m = mid_img_index - 1; m >= 0; --m) {

//    	all_images[m].finalShift(all_shifts[k].x + eskiTotalX, all_shifts[k].y + eskiTotalY);

		char str[12];
		sprintf(str, "m%d.png", m);
		char path[80]="/home/kca/Desktop/shiftedGray";
		strcat(path, str);

		int x_shift=all_shifts[m].x + eskiTotalX;
		int y_shift=all_shifts[m].y + eskiTotalY; //TODO check those

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

		RGB_shift_Image<<<dimGrid, dimBlock, 0, streams[m]>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);
//		RGB_shift_Image<<<dimGrid, dimBlock>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);

        eskiTotalX += all_shifts[m].x;
        eskiTotalY += all_shifts[m].y;
        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }

    //cout << "ikinci part baslar ..." << endl;
//    all_shifts.clear();

    //cout << " ikinci parttaki imajlari shiftliyoruz tek tek ..." << endl;

    eskiTotalX = 0;
    eskiTotalY = 0;
    for (int m = mid_img_index + 1; m < img_count; ++m) {
        //all_images[m].finalShift(all_shifts[k].x + eskiTotalX, all_shifts[k].y + eskiTotalY);

    	char str[12];
		sprintf(str, "m%d.png", m);
		char path[80]="/home/kca/Desktop/shiftedGray";
		strcat(path, str);

		int x_shift=all_shifts[m].x + eskiTotalX;
		int y_shift=all_shifts[m].y + eskiTotalY; //TODO check those

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

		RGB_shift_Image<<<dimGrid, dimBlock, 0, streams[m]>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);
		//RGB_shift_Image<<<dimGrid, dimBlock>>>(shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight, x_shift, y_shift, j_x, i_y, j_width , i_height);

        eskiTotalX += all_shifts[m].x;
        eskiTotalY += all_shifts[m].y;
        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<milliseconds<<endl;

	cudaDeviceSynchronize();

    //print original grayscale img
//	stbi_write_png(path, width, height, 3, rgb_images[mid_img_index], width*3);

	int tmpNImageSize = height * width;
	int tmpHeight = height;
	int tmpWidth = width;

	for (int var = 0; var < img_count; ++var) {

		if(var == mid_img_index){
			continue;
		}

		char str[12];
		sprintf(str, "%d.png",var);
		char path[80]="/home/kca/Desktop/img";
		strcat(path, str);


		uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize*3);
		cudaMemcpy(tmpmtb, shifted_rgb_images[var], sizeof(uint8_t)*tmpNImageSize*3, cudaMemcpyDeviceToHost);

		stbi_write_png(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth*3);
	}
	char str[12];
	sprintf(str, "%d.png",mid_img_index);
	char path[80]="/home/kca/Desktop/img";
	strcat(path, str);


	uint8_t* tmpmtb = (uint8_t *)malloc(sizeof(uint8_t)*tmpNImageSize*3);
	cudaMemcpy(tmpmtb, d_rgb_images[mid_img_index], sizeof(uint8_t)*tmpNImageSize*3, cudaMemcpyDeviceToHost);

	stbi_write_png(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth*3);



	//printf("Done..........\n");

	//stbi_write_png("/home/kca/Desktop/test1.png", width, height, 1, gray_images[0], width);
	//stbi_write_png("/home/kca/Desktop/test2.png", width, height, 1, gray_images[1], width);
	//stbi_write_png("/home/kca/Desktop/test3.png", width, height, 1, gray_images[2], width);
	//stbi_write_png("/home/kca/Desktop/test4.png", width, height, 1, gray_images[3], width);
	//stbi_write_png("/home/kca/Desktop/test5.png", width, height, 1, gray_images[4], width);

	return 0;
}
