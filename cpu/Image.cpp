//
// Created by berkay on 15.12.2018.
//

#ifndef __IMAGE_CPP__
#define __IMAGE_CPP__
#include <string.h>

#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb-master/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb-master/stb_image_resize.h"


Image::Image(char *filename) {
    read_Img(filename);
    convert2_grayscale();
    find_median();
    find_MTB_EBM();
    write_all();
}

void Image::read_Img(char* filename) {
    img = stbi_load(filename, &width, &height, &bpp, 3);

    if(img){ std::cout << filename << " Read Successfully\n"; }
    else{ std::cout << filename << " Reading Failed\n"; }
}

void Image::convert2_grayscale() {

    GRAY=(PIXEL*)malloc(width*height*sizeof(PIXEL));

    memset(hist,0,256*sizeof(int));

    int index=0;
    int gray_index=0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            GRAY[gray_index]=(54*img[index+0] + 183*img[index+1] + 19*img[index+2]) / 256.0f;
            hist[GRAY[gray_index]]++;

            gray_index++;
            index+=3;
        }
    }
}

void Image::find_median() {

    int half_way = width * height / 2;
    int sum=0;
    for (int k = 0; k < COLOR ; k++) {
        sum += hist[k];
        if(sum > half_way)
        {
            median = k;
            break;
        }
    }
}

void Image::find_MTB_EBM() {
    MTB=(PIXEL*)malloc(width*height*sizeof(PIXEL));
    EBM=(PIXEL*)malloc(width*height*sizeof(PIXEL));

    int index=0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            if (GRAY[index] < (median - 4) || GRAY[index] > (median + 4)) {

                EBM[index] = 255;
            } else {
                EBM[index] = 0;
            }

            if (GRAY[index] < median) {

                MTB[index] = 0;
            } else {
                MTB[index] = 255;
            }

            index++;
        }
    }
}

void Image::write_all() {
    stbi_write_png("gray.png", width, height, 1, GRAY, width);
    stbi_write_png("mtb.png", width, height, 1, MTB, width);
    stbi_write_png("exclusion.png", width, height, 1, EBM, width);
}

#endif