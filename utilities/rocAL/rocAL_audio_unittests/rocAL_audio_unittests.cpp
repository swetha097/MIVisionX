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
#include <vector>

#include "rocal_api.h"

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
#endif

#define DISPLAY 1

using namespace std::chrono;

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: image_augmentation <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    unsigned test_case;
    float sample_rate = 0.0; //atoi(argv[++argIdx]);
    bool downmix = false; //atoi(argv[++argIdx]);
    unsigned max_frames = 1; //atoi(argv[++argIdx]);
    unsigned max_channels = 1;


    bool gpu = 0;
    // int test_case = 3; // To be introduced later
    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

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
    int inputBatchSize = 1;
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

    RocalTensor input1, output;
    input1 = rocalAudioFileSource(handle, path, num_threads, false, false, false, max_frames, downmix);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    switch (test_case)
    {
        case 0:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalToDecibels(handle, input1, tensorLayout, tensorOutputType, true);
            std::cerr<<"\n Calls rocalToDecibels";
        }
        break;
        case 1:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalPreEmphasisFilter(handle, input1, tensorOutputType, true);
            std::cerr<<"\n Calls rocalPreEmphasisFilter ";
        }
        break;
        case 2:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            int nfftSize = 2048;
            output = rocalSpectrogram(handle, input1, tensorOutputType, true, true, true, RocalSpectrogramLayout(0), 2, nfftSize, 512, 256);
            std::cerr<<"\n Calls rocalSpectrogram ";
        }
        break;
        case 3:
        {
            output = rocalNonSilentRegion(handle, input1, true, -60, 1, -1, 3);
        }
        break;
        case 4:
        {
            std::cerr<<"\n Mel Filter Bank";
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            RocalTensor temp_output = rocalSpectrogram(handle, input1, tensorOutputType, false, true, true);
            float sampleRate = 16000;
            float minFreq = 0.0;
            float maxFreq = sampleRate / 2;
            RocalMelScaleFormula melFormula = RocalMelScaleFormula::SLANEY;
            int numFilter = 128;
            bool normalize = true;

            output = rocalMelFilterBank(handle, temp_output, true, maxFreq, minFreq, melFormula, numFilter, normalize, sampleRate);
        }
        break;
        case 5:
        {
            std::cerr<<"\n Slice";
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            const size_t num_values = 3;

            float anchor_values[num_values] = {10, 50, 100};
            double anchor_frequencies[num_values] = {1, 5, 5};
            RocalIntParam anchor = rocalCreateFloatRand(anchor_values, anchor_frequencies,
                                                    sizeof(anchor_values) / sizeof(anchor_values[0]));

            float values[num_values] = {20, 100, 200};
            double frequencies[num_values] = {1, 5, 5};
            RocalIntParam shape = rocalCreateFloatRand(values, frequencies,
                                                    sizeof(values) / sizeof(values[0]));

            float fill_values[num_values] = {10, 50, 100};
            double fill_frequencies[num_values] = {1, 5, 5};
            RocalFloatParam fill = rocalCreateFloatRand(fill_values, fill_frequencies,
                                                    sizeof(fill_values) / sizeof(fill_values[0]));

            output = rocalSlice(handle, input1, tensorOutputType, true, anchor, shape, fill, 0);
        }
        break;
        case 6:
        {
            std::cerr<<"\n Normalize";
            RocalTensorLayout tensorLayout;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalNormalize(handle, input1, tensorOutputType, true, false, {1});
        }
        break;


        default:
        {
            std::cout << "Not a valid pipeline type ! Exiting!\n";
            return -1;
        }

    }

    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    const unsigned number_of_cols = 1; //1920 / w;
    cv::Mat mat_output, mat_input;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;

    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    RocalTensorList output_tensor_list;

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        std::cerr<<"\n rocalGetRemainingImages:: "<<rocalGetRemainingImages(handle)<<"\t inputBatchsize:: "<<inputBatchSize  ;
        std::cerr<<"\n index "<<index;
        index++;
        if (rocalRun(handle) != 0)
            break;
        std::vector<float> audio_op;
        output_tensor_list = rocalGetOutputTensors(handle);
        std::cerr<<"*****************************Audio output**********************************\n";
        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            float * buffer = (float *)output_tensor_list->at(idx)->buffer();
            for(int n = 0; n < output_tensor_list->at(idx)->info().data_size() / 4; n++) // shobi check with Fiona
            {
                std::cerr << (float)buffer[n] << "\n";
            }
        }
        std::cerr<<"******************************************************************************\n";
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
    exit(0);
    return 0;
}
