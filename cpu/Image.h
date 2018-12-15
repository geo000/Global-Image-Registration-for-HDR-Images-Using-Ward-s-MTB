//
// Created by berkay on 15.12.2018.
//

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <iostream>

#define PIXEL uint8_t
#define COLOR 256
#define PYRAMID_LEVEL 6

class Image {

public:

    int width, height, bpp;
    PIXEL *img;
    PIXEL *MTB;
    PIXEL *EBM;
    PIXEL *GRAY;
    PIXEL *shiftedMTB;
    PIXEL *shiftedEMB;

//    PIXEL *gray_pyramid[6];
//    PIXEL *mtb_pyramid[6];
//    PIXEL *ebm_pyramid[6];

public:

    Image();

    Image(int _heigth, int _width);

    Image(char *filename);

    virtual ~Image() {
//        free(img);
//        free(MTB);
//        free(EBM);
//        free(GRAY);
    }

    PIXEL *getImg() const {
        return img;
    }

    PIXEL *getMTB() {
        return MTB;
    }

    PIXEL *getEBM() const {
        return EBM;
    }

    PIXEL *getGRAY() const {
        return GRAY;
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    int getBpp() const {
        return bpp;
    }

    bool read_Img(char *filename);

    static int find_median(int _height, int _width, const PIXEL *input);

    void convert2_grayscale();

    static void find_MTB_EBM(const PIXEL *input, PIXEL *_MTB, PIXEL *_EBM, int _height, int _width);


    void shift(int x, int y, int edge_values = 0);

    static PIXEL *apply_and(const PIXEL *left, const PIXEL *right, int height, int width);

    static int count_error(const PIXEL *input, int height, int width);

    PIXEL *operator^(const Image &input);

    bool compare_size(const Image &input);

    void write_all();

};

#endif