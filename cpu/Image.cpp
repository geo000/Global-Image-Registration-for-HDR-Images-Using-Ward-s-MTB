//
// Created by berkay on 15.12.2018.
//

#ifndef __IMAGE_CPP__
#define __IMAGE_CPP__

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "Image.h"


#define STB_IMAGE_IMPLEMENTATION

#include "../stb-master/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../stb-master/stb_image_write.h"


using namespace std;

Image::Image() {

}

Image::Image(int _heigth, int _width) {

    height = _heigth;
    width = _width;

    mtb = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    ebm = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    gray = (PIXEL *) malloc(width * height * sizeof(PIXEL));
}

Image::Image(char *filename) {
    if (read_Img(filename)) {
        convert2_grayscale();
    }
}


bool Image::read_Img(char *filename) {
    img = stbi_load(filename, &width, &height, &bpp, 3);

    if (img) {
        //std::cout << filename << " Read Successfully\n";
        return true;
    } else {
        //std::cout << filename << " Reading Failed\n";
        return false;
    }
}

void Image::convert2_grayscale() {

    gray = (PIXEL *) malloc(width * height * sizeof(PIXEL));

    int index = 0;
    int gray_index = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            gray[gray_index] = (54 * img[index + 0] + 183 * img[index + 1] + 19 * img[index + 2]) / 256.0f;

            gray_index++;
            index += 3;
        }
    }
}

int Image::find_median(int _height, int _width, const PIXEL *input) {

    int median;

    int hist[COLOR];
    memset(hist, 0, COLOR * sizeof(int));

    int index = 0;

    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            hist[input[index]]++;
            index++;
        }
    }

    int half_way = _height * _width / 2;
    int sum = 0;
    for (int k = 0; k < COLOR; k++) {
        sum += hist[k];
        if (sum > half_way) {
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
    stbi_write_png("../output/gray.png", width, height, 1, gray, width);
    stbi_write_png("../output/mtb.png", width, height, 1, mtb, width);
    stbi_write_png("../output/exclusion.png", width, height, 1, ebm, width);
}

int Image::count_error(const PIXEL *input, int height, int width) {
    int res = 0;

    int index = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (input[index] > 0)
                res++;
            index++;
        }
    }

    return res;

}

PIXEL *Image::apply_and(const PIXEL *left, const PIXEL *right, int height, int width) {

    PIXEL *res = (PIXEL *) malloc(width * height * sizeof(PIXEL));

    int index = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res[index] = left[index] & right[index];
            index++;
        }
    }

    return res;

}

PIXEL *Image::operator^(const Image &input) {

    if (compare_size(input)) {
        PIXEL *res = (PIXEL *) malloc(width * height * sizeof(PIXEL));

        int index = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                res[index] = this->mtb[index] ^ input.shiftedMtb[index];
                index++;
            }
        }

        return res;

    } else {
        std::cout << "DIMENSIONS NOT EQUAL" << std::endl;
        return NULL;
    }
}

void Image::shift(int x, int y, int edge_values) {

    if (x == 0 && y == 0) return;

    PIXEL *tmp = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    memset(tmp, edge_values, width * height * sizeof(PIXEL));
    PIXEL *tmp2 = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    memset(tmp2, edge_values, width * height * sizeof(PIXEL));

    //both of them are positive
    if (x >= 0 && y >= 0) {
        for (int i = 0; i < height - y; ++i) {
            for (int j = 0; j < width - x; ++j) {
                tmp[y * width + x + i * width + j] = this->mtb[i * width + j];
                tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
        return;
    }

    //both of them are negative
    if (x < 0 && y < 0) {
        for (int i = -y; i < height; ++i) {
            for (int j = -x; j < width; ++j) {
                tmp[y * width + x + i * width + j] = this->mtb[i * width + j];
                tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }


    //x neg y pos
    if (x < 0 && y >= 0) {
        for (int i = 0; i < height - y; ++i) {
            for (int j = -x; j < width; ++j) {
                tmp[y * width + x + i * width + j] = this->mtb[i * width + j];
                tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }


    //x pos y neg
    if (x >= 0 && y < 0) {
        for (int i = -y; i < height; ++i) {
            for (int j = 0; j < width - x; ++j) {
                tmp[y * width + x + i * width + j] = this->mtb[i * width + j];
                tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }

    if (!shiftedMtb) {
        delete shiftedMtb;
        delete shiftedEbm;
        shiftedMtb = NULL;
        shiftedEbm = NULL;
    }
    shiftedMtb = tmp;
    shiftedEbm = tmp2;

}


void Image::finalShift(int x, int y, int edge_values) {

    if (x == 0 && y == 0) return;

    PIXEL *tmp = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    memset(tmp, edge_values, width * height * sizeof(PIXEL));
    //PIXEL *tmp2 = (PIXEL *) malloc(width * height * sizeof(PIXEL));
    //memset(tmp2, edge_values, width * height * sizeof(PIXEL));

    //both of them are positive
    if (x >= 0 && y >= 0) {
        for (int i = 0; i < height - y; ++i) {
            for (int j = 0; j < (width - x); j++) {
                tmp[y * width + x + i * width + j] = this->gray[i * width + j];
                //tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
        return;
    }

    //both of them are negative
    if (x < 0 && y < 0) {
        for (int i = -y; i < height; ++i) {
            for (int j = -x; j < width; j++) {
                tmp[y * width + x + i * width + j] = this->gray[i * width + j];
                //tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }


    //x neg y pos
    if (x < 0 && y >= 0) {
        for (int i = 0; i < height - y; ++i) {
            for (int j = -x; j < width; j++) {
                tmp[y * width + x + i * width + j] = this->gray[i * width + j];
                //tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }


    //x pos y neg
    if (x >= 0 && y < 0) {
        for (int i = -y; i < height; ++i) {
            for (int j = 0; j < (width - x); j++) {
                tmp[y * width + x + i * width + j] = this->gray[i * width + j];
                //tmp2[y * width + x + i * width + j] = this->ebm[i * width + j];
            }
        }
    }

    if (!shiftedImg) {
        delete shiftedImg;
        shiftedImg = NULL;
    }
    shiftedImg = tmp;
}

bool Image::compare_size(const Image &input) {
    return this->height == input.height && this->width == input.width;
}


#endif