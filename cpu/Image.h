//
// Created by berkay on 15.12.2018.
//

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <iostream>

#define PIXEL uint8_t
#define COLOR 256
#define PYRAMID_LEVEL 6

class Image
{

private:

    int width, height, bpp;
    PIXEL* img;
    PIXEL* MTB;
    PIXEL* EBM;
    PIXEL* GRAY;

    PIXEL* gray_pyramid[6];
    PIXEL* mtb_pyramid[6];
    PIXEL* ebm_pyramid[6];

public:

    Image(char* filename);

    virtual ~Image() {
        free(img);
        free(MTB);
        free(EBM);
        free(GRAY);
    }

    PIXEL *getImg() const {
        return img;
    }

    PIXEL *getMTB() const {
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

    bool read_Img(char* filename);

    int find_median(int _height, int _width, const PIXEL* input);

    void convert2_grayscale();

    void find_MTB_EBM(const PIXEL *input, PIXEL *_MTB, PIXEL *_EBM, int _height, int _width);

    void write_all();

    PIXEL* operator&(const Image& input);
    PIXEL* operator^(const Image& input);
    PIXEL* operator|(const Image& input);

    bool compare_size(const Image& input);

    void make_pyramid();
};
#endif