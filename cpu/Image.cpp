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
    if(read_Img(filename))
    {
        convert2_grayscale();
        MTB = (PIXEL *) malloc(width * height * sizeof(PIXEL));
        EBM = (PIXEL *) malloc(width * height * sizeof(PIXEL));
        find_MTB_EBM(GRAY,MTB,EBM,height,width);
        write_all();
    }
}

bool Image::read_Img(char* filename) {
    img = stbi_load(filename, &width, &height, &bpp, 3);

    if(img){ std::cout << filename << " Read Successfully\n"; return true;}
    else{ std::cout << filename << " Reading Failed\n"; return false;}
}

void Image::convert2_grayscale() {

    GRAY=(PIXEL*)malloc(width*height*sizeof(PIXEL));

    int index=0;
    int gray_index=0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            GRAY[gray_index]=(54*img[index+0] + 183*img[index+1] + 19*img[index+2]) / 256.0f;

            gray_index++;
            index+=3;
        }
    }
}

int Image::find_median(int _height, int _width, const PIXEL* input) {

    int median;

    int hist[COLOR];
    memset(hist,0,COLOR*sizeof(int));

    int index=0;

    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            hist[input[index]]++;
            index++;
        }
    }

    int half_way = _height * _width / 2;
    int sum=0;
    for (int k = 0; k < COLOR ; k++) {
        sum += hist[k];
        if(sum > half_way)
        {
            median = k;
            return median;
        }
    }
}

void Image::find_MTB_EBM(const PIXEL *input, PIXEL *_MTB, PIXEL *_EBM, int _height, int _width) {

    int median = find_median(_height, _width, input);

    int index = 0;
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {

            if (input[index] < (median - 4) || input[index] > (median + 4)) {

                _EBM[index] = 255;
            } else {
                _EBM[index] = 0;
            }

            if (input[index] < median) {

                _MTB[index] = 0;
            } else {
                _MTB[index] = 255;
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

PIXEL* Image::operator&(const Image &input) {

    if(compare_size(input))
    {
        PIXEL *res = (PIXEL *) malloc(width * height * sizeof(PIXEL));

        int index=0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                res[index] = this->MTB[index] & input.MTB[index];
                index++;
            }
        }

        return res;

    } else
    {
        std::cout<<"DIMENSIONS NOT EQUAL"<<std::endl;
        return NULL;
    }
}

PIXEL* Image::operator^(const Image &input) {

    if(compare_size(input))
    {
        PIXEL *res = (PIXEL *) malloc(width * height * sizeof(PIXEL));

        int index=0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                res[index] = this->MTB[index] ^ input.MTB[index];
                index++;
            }
        }

        return res;

    } else
    {
        std::cout<<"DIMENSIONS NOT EQUAL"<<std::endl;
        return NULL;
    }
}

PIXEL* Image::operator|(const Image &input) {

    if(compare_size(input))
    {
        PIXEL *res = (PIXEL *) malloc(width * height * sizeof(PIXEL));

        int index=0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                res[index] = this->MTB[index] | input.MTB[index];
                index++;
            }
        }

        return res;

    } else
    {
        std::cout<<"DIMENSIONS NOT EQUAL"<<std::endl;
        return NULL;
    }
}

bool Image::compare_size(const Image &input) {
    return this->height != input.getHeight() && this->width != input.getWidth();
}

void Image::make_pyramid() {

    int _height = height, _width = width;
    gray_pyramid[0] = GRAY;
    mtb_pyramid[0] = MTB;
    ebm_pyramid[0] = EBM;

    for (int i = 1; i < 6; i++) {

        gray_pyramid[i] = (PIXEL *) malloc((_height/2) * (_width/2) * sizeof(PIXEL));

        stbir_resize_uint8(gray_pyramid[i-1] , _width , _height , 0,
                           gray_pyramid[i], (_width/2), (_height/2), 0, 1);

        _height/=2;
        _width/=2;

        mtb_pyramid[i] = (PIXEL *) malloc(_height * _width * sizeof(PIXEL));
        ebm_pyramid[i] = (PIXEL *) malloc(_height * _width * sizeof(PIXEL));
        find_MTB_EBM(gray_pyramid[i],mtb_pyramid[i],ebm_pyramid[i],_height,_width);
    }
}
#endif