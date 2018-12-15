//
// Created by berkay on 15.12.2018.
//

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <iostream>

#define PIXEL uint8_t
#define COLOR 256

class Image
{

private:

    int median;
    int width, height, bpp;
    PIXEL* img;
    PIXEL* MTB;
    PIXEL* EBM;
    PIXEL* GRAY;
    int hist[COLOR];

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

    int getMedian() const {
        return median;
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

    void read_Img(char* filename);
    void find_median();

    void convert2_grayscale();

    void find_MTB_EBM();

    void write_all();
};
#endif