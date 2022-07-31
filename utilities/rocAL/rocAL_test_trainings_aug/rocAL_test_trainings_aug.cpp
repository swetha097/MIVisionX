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
#include <vector>
#include<string>

#include "rocal_api.h"
#define CLASSIFICATION 1
// #define SSD 1

#include "opencv2/opencv.hpp"
using namespace cv;
#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define cvDestroyWindow destroyWindow
#endif
#define DISPLAY 1

using namespace std::chrono;

int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        std::cout <<  "Usage: rocAL_basic_test <image_dataset_folder> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts \n";
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    int rgb = 1;// process color images
    int decode_width = 0;
    int decode_height = 0;
    int test_case = 0;
    bool processing_device = 0;
    size_t decode_shard_counts = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_shard_counts = atoi(argv[++argIdx]);


    int inputBatchSize = 1;

    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }
    rocalSetSeed(0);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor input1;
    RocalMetaData metadata_output;
    RocalTensor image1, image2;

#if CLASSIFICATION
    std::cout << ">>>>>>> Running CLASSIFICATION" << std::endl;
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    input1 = rocalFusedJpegCropSingleShard(handle, folderPath1,  color_format, 0, 1, true, false, false,
                                ROCAL_USE_USER_GIVEN_SIZE, 2000, 2000);    
#elif SSD

#else
    std::cout << ">>>>>>> Running IMAGE READER" << std::endl;
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if(decode_height <= 0 || decode_width <= 0)
        input1 = rocalJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false);
    else
        input1 = rocalJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false, false,
                                    ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height);
#endif
    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : "<<rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
#if CLASSIFICATION
    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;
    std::vector<float> mean{0.485, 0.456, 0.406};
    std::vector<float> sdev{0.229, 0.224, 0.225};

    image1 = rocalResize(handle, input1, tensorLayout, tensorOutputType, 3, 224 , 224, 0, true);
    image2 = rocalCropMirrorNormalize(handle, image1, tensorLayout, tensorOutputType, 3, 224, 224, 0, 0, 0, mean, sdev, true);
#elif SSD
#else
    image1 = rocalBrightness(handle, input1, true);
#endif

    rocalVerify(handle);
    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/

    cv::Mat mat_color;
    int col_counter = 0;
    char * outName = "out_";
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    std::cerr << "Going to process images\n";
    std::cerr << "Remaining images" << rocalGetRemainingImages(handle) << "\n";
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    RocalTensorList output_tensor_list;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ?  CV_8UC3 : CV_8UC1);

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
#if CLASSIFICATION
        RocalTensorList labels = rocalGetImageLabels(handle);

        for(int i = 0; i < labels->size(); i++)
        {
            int * labels_buffer = (int *)(labels->at(i)->buffer());
            std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
        }
#elif SSD
#else
        RocalTensorList labels = rocalGetImageLabels(handle);

        for(int i = 0; i < labels->size(); i++)
        {
            int * labels_buffer = (int *)(labels->at(i)->buffer());
            std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
        }
#endif

        output_tensor_list = rocalGetOutputTensors(handle);
        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);  
#if DISPLAY
        cv::Mat mat_input;
        cv::Mat mat_output;
        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            int h = output_tensor_list->at(idx)->info().max_dims().at(1) * output_tensor_list->at(idx)->info().dims().at(0);
            int w = output_tensor_list->at(idx)->info().max_dims().at(0);
            mat_input = cv::Mat(h, w, cv_color_format);
            mat_output = cv::Mat(h, w, cv_color_format);

            if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
            {
                unsigned char *out_buffer;
                out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size());
                output_tensor_list->at(idx)->copy_data(out_buffer, false);
                mat_input.data = (unsigned char *)out_buffer;
            }
            else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                mat_input.data = (unsigned char *)(output_tensor_list->at(idx)->buffer());
            mat_input.copyTo(mat_output(cv::Rect(0, 0, w, h)));

            std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
            out_filename = std::string(outName) + std::to_string(index) + std::to_string(idx) + ".png";   // in case the user specifies non png filename

            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
            {
                cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                cv::imwrite(out_filename, mat_color, compression_params);
            }
            else
            {
                cv::imwrite(out_filename, mat_output, compression_params);
            }
        }
        mat_input.release();
        mat_output.release();
#endif
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "\nLoad     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    return 0;
}
