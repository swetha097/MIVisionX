/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <boost/filesystem.hpp>
#include <dirent.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <tuple>
#ifdef RALI_VIDEO
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#endif
#include "commons.h"

namespace filesys = boost::filesystem;

#ifdef RALI_VIDEO
typedef struct VideoProperties
{
    unsigned width, height, videos_count;
    unsigned frame_rate = 0;
    std::vector<size_t> frames_count;
    std::vector<std::string> video_file_names;
    std::vector<std::tuple<unsigned, unsigned>> start_end_frame_num;
    std::vector<std::tuple<float, float>> start_end_timestamps;
    std::vector<int> labels;
} VideoProperties;

void substring_extraction(std::string const &str, const char delim, std::vector<std::string> &out);
std::vector<unsigned> open_video_context(const char *video_file_path);
VideoProperties get_video_properties_from_txt_file(const char *file_path, bool file_list_frame_num);
VideoProperties find_video_properties(const char *source_path, bool file_list_frame_num);
#endif
