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
#include <dirent.h>
#include <vector>
#include <memory>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "ffmpeg_video_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "video_loader_module.h"

class VideoReadAndDecode
{
public:
    VideoReadAndDecode();
    ~VideoReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, VideoDecoderConfig decoder_config, int batch_size);

    //! Loads a decompressed batch of images into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded image samples
    /// \param names User's buffer provided to be filled with name of the images decoded
    /// \param max_decoded_width User's buffer maximum width per decoded image. User expects the decoder to downscale the image if image's original width is bigger than max_width
    /// \param max_decoded_height user's buffer maximum height per decoded image. User expects the decoder to downscale the image if image's original height is bigger than max_height
    /// \param roi_width is set by the load() function tp the width of the region that decoded image is located. It's less than max_width and is either equal to the original image width if original image width is smaller than max_width or downscaled if necessary to fit the max_width criterion.
    /// \param roi_height  is set by the load() function tp the width of the region that decoded image is located.It's less than max_height and is either equal to the original image height if original image height is smaller than max_height or downscaled if necessary to fit the max_height criterion.
    /// \param output_color_format defines what color format user expects decoder to decode images into if capable of doing so supported is
    VideoLoaderModuleStatus load(
            unsigned char* buff,
            std::vector<std::string>& names,
            const size_t  max_decoded_width,
            const size_t max_decoded_height,
            std::vector<uint32_t> &roi_width,
            std::vector<uint32_t> &roi_height,
            std::vector<uint32_t> &actual_width,
            std::vector<uint32_t> &actual_height,
            RaliColorFormat output_color_format);

    //! returns timing info or other status information
    Timing timing();

private:
    std::vector<std::shared_ptr<VideoDecoder>> _video_decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<unsigned char> _compressed_buff;
    std::vector<std::string> _video_names; // based on video_count
    std::map<std::string, int> _video_file_name_map;
    size_t _compressed_image_size;
    size_t _actual_read_size;
    unsigned char* _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_width; // check needed
    std::vector<size_t> _actual_decoded_height; // check needed
    std::vector<size_t> _original_width;// based on video_count
    std::vector<size_t> _original_height;// based on video_count
    std::vector<size_t> _video_frame_count; // based on video_count
    std::vector<size_t> _video_frame_start_idx; // add frame start index depending on the batch size and sequencelength
    // [0,10,20,0,10,20,0,10,20,30,40,50] // for 3 videos with sequence_length 10 0video -  30 frames, 1video - 25 frames, 2video - 54 
    std::vector<size_t> _video_idx; 
    // [0,0,0,1,1,1,2,2,2,2,2,2] 
    std::string _video_path;
    size_t start_frame;
    static const size_t MAX_COMPRESSED_SIZE = 1*1024*1024; // 1 Meg
    TimingDBG _file_load_time, _decode_time;
    size_t _batch_size;
    size_t _sequence_length;
    size_t _stride;
    size_t _video_count; // based on video_count
    size_t _frame_count;
    VideoDecoderConfig _video_decoder_config;
    bool decoder_keep_original;
    unsigned int _index_start_frame;
    int index;
};
