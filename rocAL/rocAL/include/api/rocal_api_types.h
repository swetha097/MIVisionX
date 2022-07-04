/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MIVISIONX_ROCAL_API_TYPES_H
#define MIVISIONX_ROCAL_API_TYPES_H

#include <cstdlib>

#ifndef ROCAL_API_CALL
#if defined(_WIN32)
#define ROCAL_API_CALL __stdcall
#else
#define ROCAL_API_CALL
#endif
#endif

#include <half.hpp>
using half_float::half;
#include "tensor.h"

typedef void * RocalFloatParam;
typedef void * RocalIntParam;
typedef void * RocalContext;
typedef void * RocalImage;
typedef rocALTensorList * RocalMetaData;
typedef rocALTensor * RocalTensor;
typedef rocALTensorList * RocalTensorList;

struct TimingInfo
{
    long long unsigned load_time;
    long long unsigned decode_time;
    long long unsigned process_time;
    long long unsigned transfer_time;
};
enum RocalStatus
{
    ROCAL_OK = 0,
    ROCAL_CONTEXT_INVALID,
    ROCAL_RUNTIME_ERROR,
    ROCAL_UPDATE_PARAMETER_FAILED,
    ROCAL_INVALID_PARAMETER_TYPE
};


enum RocalImageColor
{
    ROCAL_COLOR_RGB24 = 0,
    ROCAL_COLOR_BGR24 = 1,
    ROCAL_COLOR_U8  = 2,
    ROCAL_COLOR_RGB_PLANAR = 3,
};

enum RocalProcessMode
{
    ROCAL_PROCESS_GPU = 0,
    ROCAL_PROCESS_CPU = 1
};

enum RocalFlipAxis
{
    ROCAL_FLIP_HORIZONTAL = 0,
    ROCAL_FLIP_VERTICAL = 1
};

enum RocalImageSizeEvaluationPolicy
{
    ROCAL_USE_MAX_SIZE = 0,
    ROCAL_USE_USER_GIVEN_SIZE = 1,
    ROCAL_USE_MOST_FREQUENT_SIZE = 2,
    ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED = 3,    // use the given size only if the actual decoded size is greater than the given size
    ROCAL_USE_MAX_SIZE_RESTRICTED = 4,       // use max size if the actual decoded size is greater than max
};

enum RocalDecodeDevice
{
    ROCAL_HW_DECODE = 0,
    ROCAL_SW_DECODE = 1
};

enum RocalTensorLayout
{
    ROCAL_NHWC = 0,
    ROCAL_NCHW = 1
};

enum RocalTensorOutputType
{
    ROCAL_FP32 = 0,
    ROCAL_FP16 = 1,
    ROCAL_UINT8 = 2
};

enum RocalDecoderType
{
    ROCAL_DECODER_TJPEG = 0,
    ROCAL_DECODER_OPENCV = 1,
    ROCAL_DECODER_VIDEO_FFMPEG_SW = 2,
    ROCAL_DECODER_VIDEO_FFMPEG_HW = 3,
    ROCAL_DECODER_AUDIO_SNDFILE = 4
};


#endif //MIVISIONX_ROCAL_API_TYPES_H
