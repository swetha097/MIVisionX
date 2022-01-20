/*
MIT License

Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <vector>
#include<string>

#include "rocal_api.h"

using namespace cv;

using namespace std::chrono;

int test(int test_case, const char *path, const char *outName, int rgb, int gpu, int width, int height,int num_of_classes, int display_all);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT)
    {
        printf("Usage: rocal_unittests <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;

    int rgb = 1; // process color images
    bool gpu = 1;
    int test_case = 3; // For Rotate
    int num_of_classes = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
         num_of_classes = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
         display_all = atoi(argv[++argIdx]);

    test(test_case, path, outName, rgb, gpu, width, height, num_of_classes, display_all);

    return 0;
}

int test(int test_case, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all)
{
    size_t num_threads = 1;
    unsigned int inputBatchSize = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                             : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                             gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                             1);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

    RocalMetaData meta_data;

    meta_data = rocalCreateLabelReader(handle, path);

    RocalTensor input1;
    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
    else
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false,
                                    ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height; // height and width

    RocalTensor image1;

    switch (test_case)
    {
    case 0:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBrightness" << std::endl;
        image1 = rocalBrightness(handle, input1, true);
    }
    break;
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGamma" << std::endl;
        image1 = rocalGamma(handle, input1, true);
    }
    break;
    case 2:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Crop Mirror Normalize Tensor" << std::endl;
        image1 = rocalCropMirrorNormalize(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0, mean, sdev, true);
    }
    break;
    default:
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    printf("\n\nAugmented copies count %lu \n", rocalGetAugmentationBranchCount(handle));

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1; //1920 / w;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
        int label_id[inputBatchSize];
        int numOfClasses = 0;
        int image_name_length[inputBatchSize];
        rocalGetImageLabels(handle, label_id);
        int img_size = rocalGetImageNameLen(handle, image_name_length);
        char img_name[img_size];
        numOfClasses = num_of_classes;
        int label_one_hot_encoded[inputBatchSize * numOfClasses];
        rocalGetImageName(handle, img_name);
        // if (num_of_classes != 0)
        // {
        //     rocalGetOneHotImageLabels(handle, label_one_hot_encoded, numOfClasses);
        // }
        std::cerr << "\nPrinting image names of batch: " << img_name<<"\n";
        for (unsigned int i = 0; i < inputBatchSize; i++)
        {
            std::cerr<<"\t Printing label_id : " << label_id[i] << std::endl;
            if(num_of_classes != 0)
            {
            std::cout << "One Hot Encoded labels:"<<"\t";
            for (int j = 0; j < numOfClasses; j++)
            {
                int idx_value = label_one_hot_encoded[(i*numOfClasses)+j];
                if(idx_value == 0)
                std::cout << idx_value;
                else
                {
                    std::cout << idx_value;
                }
            }
            }
            std::cout << "\n";
        }
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
        if (display_all)
          out_filename = std::string(outName) + std::to_string(index) + ".png";   // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imwrite(out_filename, mat_color, compression_params);
        }
        else
        {
            cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    mat_output.release();
    if (!image1)
        return -1;
    return 0;
}
