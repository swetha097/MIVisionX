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

void
VideoReadAndDecode::create(ReaderConfig reader_config, VideoDecoderConfig decoder_config, int batch_size)
{
    // Can initialize it to any decoder types if needed
    // find video_count resize newly introduced video wrt video count
    //_video_count = _reader.get_video_file_count(); // check if we can use this. if not we can find the number of files/
    std::cerr<<"\n ********************** VideoReadAndDecode::create ***************************** ";
    _sequence_length = reader_config.get_sequence_length();
    std::cerr<<"\n _sequence_length ::"<<_sequence_length;
    _video_count = reader_config.get_video_count();
    std::cerr<<"\n _video_count:: "<<_video_count;
    _batch_size = batch_size;
    std::cerr<<"\n batchsize :: "<<_batch_size;
    _compressed_buff.resize(batch_size);
    _video_decoder.resize(_video_count); // It should not be for batch size but for every video in the path
    _actual_read_size.resize(batch_size);
    _video_names.resize(batch_size);
    _compressed_image_size.resize(batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _actual_decoded_width.resize(_video_count);
    _actual_decoded_height.resize(_video_count);
    _original_height.resize(_video_count);
    _original_width.resize(_video_count);
    _video_frame_count.resize(_video_count);
    std::cerr << "\n Frame count :: " <<  reader_config.get_frame_count();
    _video_frame_count[0] = reader_config.get_frame_count();
    _video_decoder_config = decoder_config;

    // get the width and height for every video _actual_decoded & original
    // fill the _video_frame_start_idx & _video_idx  based on sequence length and frame count
    // shuffle both _video_frame_start_idx & _video_idx ( can do this later)
    //for sample test
    //_video_frame_count[3] = {30, 25, 54}; 

    for(int i = 0; i < _video_count; i++)
    {
        int count_sequence = 0;
        std::cerr << "\n Frames per video : " << _video_frame_count[i];
        int loop_index;
        if(_video_frame_count[i] % _sequence_length == 0)
            loop_index = (_video_frame_count[i] / _sequence_length) - 1;
        else
            loop_index = _video_frame_count[i] / _sequence_length;
        for(int j = 0; j <= loop_index; j++)
        {
            _video_frame_start_idx.push_back(count_sequence);
            _video_idx.push_back(i);
            count_sequence = count_sequence + _sequence_length;
        }
    }
    /*std::cerr << "\nSize : " << _video_frame_start_idx.size();
    for(int i = 0; i < _video_frame_start_idx.size(); i++)
    {
        std::cerr << "\n Video start_idx : " << _video_frame_start_idx[i];
        std::cerr << "\n Video idx : " << _video_idx[i];
    }*/

    for(int i = 0; i < _video_count; i++)
    {
        _compressed_buff[i].resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
        _video_decoder[i] = create_video_decoder(decoder_config);
        // _video_decoder[i]->initialize() Is it gonna come here? 
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
                         RaliColorFormat output_color_format,
                         bool decoder_keep_original )
{
    if(max_decoded_width == 0 || max_decoded_height == 0 )
        THROW("Zero image dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count() < _batch_size)
        return VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load images/frames from the disk and push them as a large image onto the buff
    unsigned file_counter = 0;
    const auto ret = video_interpret_color_format(output_color_format);
    const VideoDecoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);
    const bool keep_original = decoder_keep_original; // check and remove

    // Decode with the height and size equal to a single image  
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing

    while ((file_counter != _batch_size) && _reader->count() > 0)
    {
        size_t fsize = 1280 * 720 * 3; //_reader->open(); Have to set the max frame size
        if (fsize == 0) {
            WRN("Opened file " + _reader->id() + " of size 0");
            continue;
        }

        _compressed_buff[file_counter].reserve(fsize);

        _actual_read_size[file_counter] = _reader->read(_compressed_buff[file_counter].data(), fsize);
        _video_names[file_counter] = _reader->id();
        _reader->close();
        _compressed_image_size[file_counter] = fsize;
        file_counter++;
    }

    _file_load_time.end();// Debug timing
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);

    for(size_t i = 0; i < _batch_size; i++)
        _decompressed_buff_ptrs[i] = buff + image_size * i;

    _decode_time.start();// Debug timing
#pragma omp parallel for num_threads(_batch_size)  // default(none) TBD: option disabled in Ubuntu 20.04
    for(size_t i= 0; i < _batch_size; i++)
    {
        // initialize the actual decoded height and width with the maximum
        _actual_decoded_width[i] = max_decoded_width;
        _actual_decoded_height[i] = max_decoded_height;
        
        int original_width, original_height, jpeg_sub_samp;
        /*if(_decoder[i]->decode_info(_compressed_buff[i].data(), _actual_read_size[i], &original_width, &original_height, &jpeg_sub_samp ) != Decoder::Status::OK)
        {
            continue;
        }*/
        _original_height[i] = original_height;
        _original_width[i]  = original_width;
#if 0
        if((unsigned)original_width != max_decoded_width || (unsigned)original_height != max_decoded_height)
            // Seeting the whole buffer to zero in case resizing to exact output dimension is not possible.
            memset(_decompressed_buff_ptrs[i],0 , image_size);
#endif

        // decode the image and get the actual decoded image width and height
        size_t scaledw, scaledh;
        std::string out = "out.mp4";
        // if we process 3rd video then we need to call _decode[2]
        _video_decoder[i]->Decode(_decompressed_buff_ptrs[i], (_video_names[i]).c_str(), out.c_str());
       
        /*if(_decoder[i]->decode(_compressed_buff[i].data(),_compressed_image_size[i],_decompressed_buff_ptrs[i],
                               max_decoded_width, max_decoded_height,
                               original_width, original_height,
                               scaledw, scaledh,
                               decoder_color_format,_decoder_config, keep_original) != Decoder::Status::OK)
        {
            continue;
        }*/
        _actual_decoded_width[i] = scaledw;
        _actual_decoded_height[i] = scaledh;
    }
    for(size_t i = 0; i < _batch_size; i++)
    {
        names[i] = _video_names[i];
        roi_width[i] = _actual_decoded_width[i];
        roi_height[i] = _actual_decoded_height[i];
        actual_width[i] = _original_width[i];
        actual_height[i] = _original_height[i];
    }

    _decode_time.end();// Debug timing

    return VideoLoaderModuleStatus::OK;
}
