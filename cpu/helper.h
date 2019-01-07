////
//// Created by kca on 15.12.2018.
////

#include "Image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "../stb-master/stb_image_resize.h"

void make_pyramid(Image **pyramid, Image &original) {

    int height = original.height, width = original.width;

    pyramid[0] = new Image(height, width);

    Image::find_MTB_EBM(original.gray, pyramid[0]->mtb, pyramid[0]->ebm, height, width);

    pyramid[0]->gray = original.gray;

    for (int i = 1; i < PYRAMID_LEVEL; i++) {

        pyramid[i] = new Image((height / 2), (width / 2));

        stbir_resize_uint8(pyramid[i - 1]->gray, width, height, 0,
                           pyramid[i]->gray, (width / 2), (height / 2), 0, 1);

        height /= 2;
        width /= 2;

        Image::find_MTB_EBM(pyramid[i]->gray, pyramid[i]->mtb, pyramid[i]->ebm, height, width);
    }
}
