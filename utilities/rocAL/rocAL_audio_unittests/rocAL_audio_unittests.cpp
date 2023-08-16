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
#define METADATA 0 // Switch the meta-data part once the meta-data reader (file list reader) is introduced
using namespace std::chrono;

void test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu, int batch_size);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: rocAL_audio_unittests <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 batch_size\n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    float sample_rate = 16000;
    bool downmix = false;
    unsigned max_frames = 1;
    unsigned max_channels = 1;
    int batch_size = 1;
    bool gpu = 0;
    unsigned test_case;
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
    
    if (argc >= argIdx + MIN_ARG_COUNT)
        batch_size = atoi(argv[++argIdx]);

    test(test_case, path, sample_rate, downmix, max_frames, max_channels, gpu, batch_size);
    return 0;
}

void test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu, int batch_size)
{
    size_t num_threads = 1;
    int inputBatchSize = batch_size;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(inputBatchSize,
                             gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                             1);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal context\n";
        return;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);


    RocalMetaData metadata_output;
    // MetaData reader for input file_list which has file seperated by labels

    // if (METADATA) { // To uncomment later when meta-data reader for audio is added
    //     std::cerr << "META DATA READER";
    //     const char* file_list_path = "/workspace/rnnt/AMD/MIVisionX-data/rocal_data/audio_samples/audio_file_list.txt" ; // TODO: Add this as an arg in main() 
    //     metadata_output = rocalCreateFileListLabelReader(handle, path, file_list_path);
    // }

    //Decoder
    RocalTensor input1, output;
    const char* file_list_path = "/media/MIVisionX-data/rocal_data/audio_samples/audio_file_list.txt" ; // use it when meta-data reader is introduced
    input1 = rocalAudioFileSourceSingleShard(handle, path, file_list_path, 0, 1, true, false, false, false, max_frames, max_channels, 0, false, -1); // Yet to give support for stick_to_shard & shard_size
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return;
    }

    switch (test_case)
    {
        case 0:
        {
            float cutOffDB = std::log(1e-20);
            float multiplier = std::log(10);
            float referenceMagnitude = 1.0f;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalToDecibels(handle, input1, true, cutOffDB, multiplier, referenceMagnitude, tensorOutputType);
            std::cerr<<"\nCalls rocalToDecibels";
            break;
        }
        case 1:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalPreEmphasisFilter(handle, input1, tensorOutputType, true);
            std::cerr<<"\nCalls rocalPreEmphasisFilter";
            break;
        }
        case 2:
        {
            std::vector<float> windowFn {};
            int power = 2;
            int nfft = 512;
            int windowLength = 320;
            int windowStep = 160;
            RocalSpectrogramLayout layout = RocalSpectrogramLayout::FT;
            bool centerWindows = true;
            bool reflectPadding = true;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalSpectrogram(handle, input1, true, windowFn, centerWindows, reflectPadding, layout, power, nfft, windowLength, windowStep, tensorOutputType);
            std::cerr<<"\nCalls rocalSpectrogram";
            break;
        }
        case 3:
        {
            float cutOffDB = -60.0;
            int windowLength = 2048;
            float referencePower = 0.0f;
            int resetInterval = 8192;
            auto output = rocalNonSilentRegion(handle, input1, false, cutOffDB, referencePower, resetInterval, windowLength);
            RocalTensor begin = output.first;
            RocalTensor length = output.second;
            std::cerr<<"\nCalls rocalNonSilentRegion";
            break;
        }
        case 4:
        {
            std::vector<float> windowFn;
            int power = 2;
            int nfft = 512;
            int windowLength = 320;
            int windowStep = 160;
            RocalSpectrogramLayout layout = RocalSpectrogramLayout::FT;
            bool centerWindows = true;
            bool reflectPadding = true;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            RocalTensor spectrogramOutput = rocalSpectrogram(handle, input1, true, windowFn, centerWindows, reflectPadding, layout, power, nfft, windowLength, windowStep, tensorOutputType);
            
            float sampleRate = 16000;
            float minFreq = 0.0;
            float maxFreq = sampleRate / 2;
            RocalMelScaleFormula melFormula = RocalMelScaleFormula::SLANEY;
            int numFilter = 128;
            bool normalize = true;
            output = rocalMelFilterBank(handle, spectrogramOutput, true, maxFreq, minFreq, melFormula, numFilter, normalize, sampleRate, tensorOutputType);
            std::cerr<<"\nCalls rocalMelFilterBank";
            break;
        }
        case 5:
        {
            float cutOffDB = -60.0;
            int windowLength = 2048;
            float referencePower = 0.0f;
            int resetInterval = 8192;
            std::pair <RocalTensor,RocalTensor>  nonSilentRegionOutput;
            nonSilentRegionOutput = rocalNonSilentRegion(handle, input1, false, cutOffDB, referencePower, resetInterval, windowLength);
            
            RocalOutOfBoundsPolicy policy = RocalOutOfBoundsPolicy::ERROR;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalSlice(handle, input1, true, nonSilentRegionOutput.first, nonSilentRegionOutput.second, {0.0f}, {0}, false, false, policy, tensorOutputType);
            std::cerr<<"\nCalls rocalSlice";
            break;
        }
        case 6:
        {
            float mean, stdDev, scale, shift, epsilon;
            mean = stdDev = scale = shift = epsilon = 0.0f;
            int ddof = 0;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalNormalize(handle, input1, true, false, {1}, mean, stdDev, scale, shift, ddof, epsilon, tensorOutputType);
            std::cerr<<"\nCalls rocalNormalize";
            break;
        }
        default:
        {
            std::cout << "Not a valid pipeline type ! Exiting!\n";
            return;
        }
    }
    
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return;
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
        if (rocalRun(handle) != 0) {
            break;
        }
        std::vector<float> audio_op;
        output_tensor_list = rocalGetOutputTensors(handle);
        std::cerr<<"\n *****************************Audio output**********************************\n";
        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            float * buffer = (float *)output_tensor_list->at(idx)->buffer();
            for(int n = 0; n < 5; n++)
            {
                std::cerr << buffer[n] << "\n";
            }
            
        }

        
        if (METADATA) {
            RocalTensorList labels = rocalGetImageLabels(handle);

            for(int i = 0; i < labels->size(); i++)
            {
                int * labels_buffer = (int *)(labels->at(i)->buffer());
                std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";


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
    rocalRelease(handle);
    exit(0);
    return;
}
