/*
MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rocal_api.h"

using namespace cv;

#define AUDIO

using namespace std::chrono;

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: image_augmentation <audio-dataset-folder> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    float sample_rate = 0.0; //atoi(argv[++argIdx]);
    bool downmix = false; //atoi(argv[++argIdx]);
    unsigned max_frames = 1; //atoi(argv[++argIdx]);
    unsigned max_channels = 1;


    bool gpu = 0;
    int test_case = 3; // To be introduced later

    if (argc >= argIdx + MIN_ARG_COUNT)
        sample_rate = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        downmix = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_frames = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_channels = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);


    int return_val = test(test_case, path, sample_rate, downmix, max_frames, max_channels, gpu);
    return 0;
}

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu)
{
    size_t num_threads = 1;
    int inputBatchSize = 2;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << std::endl;


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

    RocalTensor input1;
    input1 = rocalAudioFileSource(handle, path, num_threads, true, false, false);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    rocalVerify(handle);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    std::cout << "\n\nAugmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    const unsigned number_of_cols = 1; //1920 / w;
    cv::Mat mat_output, mat_input;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
    // switch (tensorOutputType)
    // {
    //     case ROCAL_FP32:
    //     {
    //         mat_output = cv::Mat(h, w, CV_32FC3);
    //     }
    //     break;
    //     case ROCAL_UINT8:
    //     {
    //         auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    //         mat_output = cv::Mat(h, w, cv_color_format);
    //         mat_input = cv::Mat(h, w, cv_color_format);
    //     }
    //     break;
    // }

    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        std::cerr<<"\n rocalGetRemainingImages:: "<<rocalGetRemainingImages(handle)<<"\t inputBatchsize:: "<<inputBatchSize  ;
        std::cerr<<"\n index "<<index;
        index++;
        if (rocalRun(handle) != 0)
        {
            // sleep(2);
            std::cerr<<"\n Inside rocAl run\n";
            break;
        }

        // switch (tensorOutputType)
        // {
        // case ROCAL_FP32:
        //     rocalCopyToTensorOutput(handle, (float *)mat_input.data, h * w * p);
        //     break;
        // case ROCAL_UINT8:
        //     rocalCopyToTensorOutput(handle, (unsigned char *)mat_input.data, h * w * p);
        //     break;
        // }
        // int label_id[inputBatchSize];
        // int image_name_length[inputBatchSize];
        // rocalGetImageLabels(handle, label_id);
        // int img_size = rocalGetImageNameLen(handle, image_name_length);
        // char img_name[img_size];
        // rocalGetImageName(handle, img_name);
        // std::cerr << "\nPrinting image names of batch: " << img_name;
        // if (!display)
        //     continue;

        // std::vector<int> compression_params;
        // compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        // compression_params.push_back(9);

        // mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        // if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
        // {
        //     cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
        //     //cv::imshow("output", mat_color);
        //     cv::imwrite(std::to_string(index) + outName, mat_color, compression_params);
        //     // cv::waitKey(0);
        // }
        // else {
        //     //cv::imshow("output", mat_output);
        //     cv::imwrite(std::to_string(index) + outName, mat_output, compression_params);
        // }
        // col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    // rocalRelease(handle);
    // mat_input.release();
    // mat_output.release();
    exit(0);
    return 0;
}
