//
// Created by berkay on 14.12.2018.
//
#include <vector>
#include <pthread.h>
#include "Image.h"
#include "helper.h"
#include "../stb-master/stb_image_write.h"
#include <ctime>

using namespace std;

typedef struct shift_pair {
    shift_pair() {
        x = 0;
        y = 0;
    }
    shift_pair(int _x, int _y) {
        x = _x;
        y = _y;
    }

    shift_pair(const shift_pair &rhs) {
        x = rhs.x;
        y = rhs.y;
    }

    int x;
    int y;
} shift_pair;

typedef struct arg_struct {
    shift_pair *shiftp;
    std::vector<Image> *all_images;
    int ind1;
    int ind2;
} arguments;

//shift_pair calculateOffsetOfTwoImages(std::vector<Image> &all_images, int ind1, int ind2) {
//    Image *first[PYRAMID_LEVEL];
//    Image *second[PYRAMID_LEVEL];
//
//    make_pyramid(first, all_images[ind1]);
//    make_pyramid(second, all_images[ind2]);
//
//
//    //stbi_write_png("../output/mtb.png", second[0]->width, second[0]->height, 1, second[0]->mtb, second[0]->width);
//
//
//    int curr_level = PYRAMID_LEVEL - 1;
//    int curr_offset_x = 0;
//    int curr_offset_y = 0;
//    int offset_return_x = 0;
//    int offset_return_y = 0;
//
//    for (int k = curr_level; k >= 0; --k) {
//        curr_offset_x = 2 * offset_return_x;
//        curr_offset_y = 2 * offset_return_y;
//
//        int min_error = 255 * second[k]->height * second[k]->width;
//
//        for (int i = -1; i <= 1; ++i) {
//            for (int j = -1; j <= 1; ++j) {
//                int xs = curr_offset_x + i;
//                int ys = curr_offset_y + j;
//                second[k]->shift(xs, ys);
//                PIXEL *xor_result = (*(first[k])) ^(*(second[k]));
//
//                PIXEL *after_first_and = Image::apply_and(xor_result, first[k]->ebm, first[k]->height,
//                                                          first[k]->width);
//                PIXEL *after_second_and = Image::apply_and(after_first_and, second[k]->shiftedEbm,
//                                                           second[k]->height, second[k]->width);
//
//                int error = Image::count_error(after_second_and, second[k]->height, second[k]->width);
//                if (error < min_error) {
//                    offset_return_x = xs;
//                    offset_return_y = ys;
//                    min_error = error;
//                }
//            }
//        }
//    }
//
//    //cout << "yapilmasi gereken x-y kaydirmasi: " << curr_offset_x << "  " << curr_offset_y << endl;
//
//    return shift_pair(curr_offset_x, curr_offset_y);
//}

void *calculateOffsetOfTwoImages(void *args) {

    arguments *_arg;
    _arg = (arguments *) args;

    shift_pair *shiftp = _arg->shiftp;
    std::vector<Image> *all_images = _arg->all_images;
    int ind1 = _arg->ind1;
    int ind2 = _arg->ind2;

    Image *first[PYRAMID_LEVEL];
    Image *second[PYRAMID_LEVEL];

    make_pyramid(first, (*all_images)[ind1]);
    make_pyramid(second, (*all_images)[ind2]);


    //stbi_write_png("../output/mtb.png", second[0]->width, second[0]->height, 1, second[0]->mtb, second[0]->width);


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

                PIXEL *after_first_and = Image::apply_and(xor_result, first[k]->ebm, first[k]->height,
                                                          first[k]->width);
                PIXEL *after_second_and = Image::apply_and(after_first_and, second[k]->shiftedEbm,
                                                           second[k]->height, second[k]->width);

                int error = Image::count_error(after_second_and, second[k]->height, second[k]->width);
                if (error < min_error) {
                    offset_return_x = xs;
                    offset_return_y = ys;
                    min_error = error;
                }
            }
        }
    }

    //cout << "yapilmasi gereken x-y kaydirmasi: " << curr_offset_x << "  " << curr_offset_y << endl;

    _arg->shiftp->x = curr_offset_x;
    _arg->shiftp->y = curr_offset_y;

    pthread_exit(nullptr);
}

int main(int argc, char *argv[]) {

    //CPU TIMER STARTS
    std::clock_t c_start = std::clock();

    std::vector<Image> all_images;


    for (int l = 1; l < argc; ++l) {
        all_images.emplace_back(argv[l]);
    }

    int img_count = argc - 1;

    int num_threads = img_count;
    pthread_t threads[num_threads];

    shift_pair all_shifts[img_count];

    //cout << "Total number of images provided: " << all_images.size() << endl;

    int mid_img_index = all_images.size() / 2;

    //cout << "ilk part baslar ..." << endl;

    for (int m = mid_img_index - 1; m >= 0; --m) {

        auto *args = (arguments *) malloc(sizeof(arguments));
        args->shiftp = &(all_shifts[m]);
        args->all_images = &all_images;
        args->ind1 = m + 1;
        args->ind2 = m;

        //cout << "part1 thread create ediyoz..." << endl;
        int rc = pthread_create(&threads[m], nullptr, calculateOffsetOfTwoImages, (void *) args);

        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }

        //all_shifts.emplace_back(calculateOffsetOfTwoImages(all_images, m + 1, m));
    }

    for (int m = mid_img_index + 1; m < all_images.size(); ++m) {

        auto *args = (arguments *) malloc(sizeof(arguments));
        args->shiftp = &(all_shifts[m]);
        args->all_images = &all_images;
        args->ind1 = m - 1;
        args->ind2 = m;

        //cout << "part1 thread create ediyoz..." << endl;
        int rc = pthread_create(&threads[m], nullptr, calculateOffsetOfTwoImages, (void *) args);

        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }

        //all_shifts.emplace_back(calculateOffsetOfTwoImages(all_images, m - 1, m));
    }

    void *status;
    for (int var = 0; var < num_threads; ++var) {
        if (var != mid_img_index)
            pthread_join(threads[var], &status);
    }

    //cout << " ilk parttaki imajlari shiftliyoruz tek tek ..." << endl;

    int eskiTotalX = 0, eskiTotalY = 0;
    for (int m = mid_img_index - 1; m >= 0; --m) {
        all_images[m].finalShift(all_shifts[m].x + eskiTotalX, all_shifts[m].y + eskiTotalY);
        eskiTotalX += all_shifts[m].x;
        eskiTotalY += all_shifts[m].y;

        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }

    //cout << "ikinci part baslar ..." << endl;
    //all_shifts.clear();

    //cout << " ikinci parttaki imajlari shiftliyoruz tek tek ..." << endl;

    eskiTotalX = 0;
    eskiTotalY = 0;
    for (int m = mid_img_index + 1; m < all_images.size(); ++m) {
        all_images[m].finalShift(all_shifts[m].x + eskiTotalX, all_shifts[m].y + eskiTotalY);
        eskiTotalX += all_shifts[m].x;
        eskiTotalY += all_shifts[m].y;
        //cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
    }

    // your_algorithm
    std::clock_t c_end = std::clock();

    double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";

    cout << "dosyaya yaziyoruz tum imajlari tek tek ..." << endl;

    //TODO write using loop
    stbi_write_jpg("../output/out0.jpg", all_images[0].width, all_images[0].height, 1, all_images[0].shiftedImg,
                   all_images[0].width);
    stbi_write_jpg("../output/out1.jpg", all_images[1].width, all_images[1].height, 1, all_images[1].shiftedImg,
                   all_images[1].width);
    stbi_write_jpg("../output/out2.jpg", all_images[2].width, all_images[2].height, 1, all_images[2].gray,
                   all_images[2].width);
    stbi_write_jpg("../output/out3.jpg", all_images[3].width, all_images[3].height, 1, all_images[3].shiftedImg,
                   all_images[3].width);
    stbi_write_jpg("../output/out4.jpg", all_images[4].width, all_images[4].height, 1, all_images[4].shiftedImg,
                   all_images[4].width);

    return 0;
}
