//
// Created by berkay on 14.12.2018.
//
#include "Image.h"
#include "helper.h"

using namespace std;

int main(int argc, char *argv[]) {
    Image temp("../input/1.JPG");
    Image temp2("../input/3.JPG");

    Image *first[PYRAMID_LEVEL];
    Image *second[PYRAMID_LEVEL];

    make_pyramid(first, temp);
    make_pyramid(second, temp2);

    int curr_level = PYRAMID_LEVEL - 1;
    int curr_offset_x = 0;
    int curr_offset_y = 0;
    int offset_return_x = 0;
    int offset_return_y = 0;

    for (int k = curr_level; k >= 0; --k) {
        curr_offset_x = 2 * offset_return_x;
        curr_offset_y = 2 * offset_return_y;

        int min_error = 255 * second[k]->height * second[k]->width;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int xs = curr_offset_x + i;
                int ys = curr_offset_y + j;
                second[k]->shift(xs, ys);
                PIXEL *xor_result = (*(first[k])) ^(*(second[k]));
//
//                Image a(first[k]->height, first[k]->width);
//                a.MTB = first[k]->MTB;
//                std::string name("../output/mtb1_result");
//                name+=std::to_string(k);
//                name += ".png";
//                a.dataYaz(name.c_str());
//                a.MTB=NULL;
//
//                Image b(second[k]->height, second[k]->width);
//                b.MTB = second[k]->MTB;
//                std::string name2("../output/mtb2_result");
//                name2+=std::to_string(k);
//                name2 += ".png";
//                b.dataYaz(name2.c_str());
//                b.MTB=NULL;


                PIXEL *after_first_and = Image::apply_and(xor_result, first[k]->EBM, first[k]->height,
                                                          first[k]->width);
                PIXEL *after_second_and = Image::apply_and(after_first_and, second[k]->shiftedEMB,
                                                           second[k]->height, second[k]->width);

                int error = Image::count_error(after_second_and, second[k]->height, second[k]->width);
                if (error < min_error) {
                    offset_return_x = xs;
                    offset_return_y = ys;
                    min_error = error;
                }

//                delete[] xor_result;
//                delete[] after_first_and;
//                delete[] after_second_and;

            }

        }
    }
//
    cout << "yapilmasi gereken x-y kaydirmasi: " << curr_offset_x << "  " << curr_offset_y << endl;
//
//    cout<<"shiftliyoz..."<<endl;
//
//    temp2.shift2(curr_offset_x, curr_offset_y);

    return 0;
}
