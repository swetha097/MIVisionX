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


#include <iterator>
#include <cstring>
#include <map>
#include "video_decoder_factory.h"
#include "video_read_and_decode.h"

namespace filesys = boost::filesystem;

std::tuple<VideoDecoder::ColorFormat, unsigned >
video_interpret_color_format(RaliColorFormat color_format )
{
    switch (color_format) {
        case RaliColorFormat::RGB24:
            return  std::make_tuple(VideoDecoder::ColorFormat::RGB, 3);

        case RaliColorFormat::BGR24:
            return  std::make_tuple(VideoDecoder::ColorFormat::BGR, 3);

        case RaliColorFormat::U8:
            return  std::make_tuple(VideoDecoder::ColorFormat::GRAY, 1);

        default:
            throw std::invalid_argument("Invalid color format\n");
    }
}

Timing
VideoReadAndDecode::timing()
{
    Timing t;
    t.image_decode_time = _decode_time.get_timing();
    t.image_read_time = _file_load_time.get_timing();
    t.shuffle_time = _reader->get_shuffle_time();
    return t;
}

VideoReadAndDecode::VideoReadAndDecode():
    _file_load_time("FileLoadTime", DBG_TIMING ),
    _decode_time("DecodeTime", DBG_TIMING)
{
}

VideoReadAndDecode::~VideoReadAndDecode()
{
    _reader = nullptr;
    _video_decoder.clear();
}

void substring_extraction(std::string const &str, const char delim,  std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

void
VideoReadAndDecode::create(ReaderConfig reader_config, VideoDecoderConfig decoder_config, int batch_size)
{

    // std::cerr<<"\n ********************** VideoReadAndDecode::create ***************************** ";
    _sequence_length = reader_config.get_sequence_length();
    _stride = reader_config.get_frame_stride();
    _video_count = reader_config.get_video_count();
    _video_names = reader_config.get_video_file_names();
    _batch_size = batch_size;
    set_video_process_count(_video_count);
    std::cerr<<"\n _sequence_length ::"<<_sequence_length;
    // std::cerr<<"\n _video_count:: "<<_video_count;
    // std::cerr<<"\n batchsize :: "<<_batch_size;

    _video_decoder.resize(_video_process_count);
    _video_names.resize(_video_count);
    _actual_decoded_width.resize(_sequence_length);
    _actual_decoded_height.resize(_sequence_length);
    _original_height.resize(_sequence_length);
    _original_width.resize(_sequence_length);

    _video_decoder_config = decoder_config;
    _index_start_frame = 0;

    _compressed_buff.resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
    size_t i=0;
    for(; i < _video_process_count; i++)
    {
        _video_decoder[i] = create_video_decoder(decoder_config);
        std::vector<std::string> substrings;
        char delim = '#';
        substring_extraction(_video_names[i], delim, substrings);

        video_map video_instance;
        video_instance._video_map_idx  = atoi(substrings[0].c_str());
        video_instance._is_decoder_instance = false;
        _video_file_name_map.insert(std::pair<std::string, video_map>(_video_names[i], video_instance));
        _video_decoder[i]->Initialize(substrings[1].c_str());

    }
    if(_video_process_count != _video_count)
    {
        while(i < _video_count)
        {
            std::vector<std::string> substrings;
            char delim = '#';
            substring_extraction(_video_names[i], delim, substrings);
            video_map video_instance;
            video_instance._video_map_idx  = atoi(substrings[0].c_str());
            video_instance._is_decoder_instance = false;
            _video_file_name_map.insert(std::pair<std::string, video_map>(_video_names[i], video_instance));
            i++;
        }
    }
    _reader = create_reader(reader_config);
}

void
VideoReadAndDecode::reset()
{
    // TODO: Reload images from the folder if needed
    _reader->reset();
}

size_t
VideoReadAndDecode::count()
{
    return _reader->count();
}

VideoLoaderModuleStatus
VideoReadAndDecode::load(unsigned char* buff,
                         std::vector<std::string>& names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         std::vector<uint32_t> &actual_width,
                         std::vector<uint32_t> &actual_height,
                         RaliColorFormat output_color_format)
{
    // std::cerr << "\nHey is comes to load!!!!! - > " << _index_start_frame;
    if(max_decoded_width == 0 || max_decoded_height == 0 )
        THROW("Zero image dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count() < _batch_size)
        return VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;

    const auto ret = video_interpret_color_format(output_color_format);
    const VideoDecoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);
    int video_idx_map;
    // Decode with the height and size equal to a single image
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing

    size_t fsize = 1280 * 720 * 3;
    if (fsize == 0) {
        WRN("Opened file " + _reader->id() + " of size 0");
    }

    start_frame = _reader->read(_compressed_buff.data(), fsize);
    _video_path = _reader->id();
    _reader->close();

    _file_load_time.end();// Debug timing
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);

    _decompressed_buff_ptrs = buff;

    _decode_time.start();// Debug timing
// #pragma omp parallel for num_threads(_batch_size)  // default(none) TBD: option disabled in Ubuntu 20.04

    for(int i = 0; i < 1; i++)
    {
        for(size_t s = 0; s < _sequence_length; s++)
        {
            _actual_decoded_width[s] = max_decoded_width;
            _actual_decoded_height[s] = max_decoded_height;
        }

        // std::cerr << "\nThe source video is " << _video_path << " MAP : "<<_video_file_name_map[_video_path]<< "\tThe start index is : " << start_frame << "\n";
        // video_idx_map = _video_file_name_map[_video_path];
        itr = _video_file_name_map.find(_video_path);
        if (itr->second._is_decoder_instance == false)
        {
            std::map<std::string, video_map>::iterator temp_itr;
            for(temp_itr = _video_file_name_map.begin(); temp_itr != _video_file_name_map.end(); ++temp_itr)
            {
                if(temp_itr->second._is_decoder_instance == true)
                {
                    video_idx_map = temp_itr->second._video_map_idx;
                    std::vector<std::string> substrings;
                    char delim = '#';
                    substring_extraction(itr->first, delim, substrings);
                    int map_idx = atoi(substrings[0].c_str());
                    _video_decoder[video_idx_map]->Initialize(substrings[1].c_str());
                    itr->second._video_map_idx = video_idx_map;
                    itr->second._is_decoder_instance = true;
                    temp_itr->second._is_decoder_instance = false;
                }
            }
        }
        video_idx_map = itr->second._is_decoder_instance;
        if(_video_decoder[video_idx_map]->Decode(_decompressed_buff_ptrs, start_frame, _sequence_length, _stride) != VideoDecoder::Status::OK)
        {
            continue;
        }
    }

    std::vector<std::string> substrings;
    char delim = '/';
    substring_extraction(_video_path, delim, substrings);

    std::string file_name = substrings[substrings.size()- 1];
    delim = '#';
    substring_extraction(_video_path, delim, substrings);
    std::string video_idx = substrings[0];
    for(size_t i = 0; i < _sequence_length ; i++)
    {
        names[i] =  video_idx   + "#" + file_name +"_"+  std::to_string(start_frame+ (i*_stride));
        // std::cerr<<"\n name: "<<names[i];
        roi_width[i] = _actual_decoded_width[i];
        roi_height[i] = _actual_decoded_height[i];
    }

    ++ _index_start_frame;
    _decode_time.end();// Debug timing

    return VideoLoaderModuleStatus::OK;
}
