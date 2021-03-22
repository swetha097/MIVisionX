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

#include <memory>
#include <sstream>
#include <vx_amd_media.h>
#include <vx_ext_amd.h>
#include <vx_ext_rpp.h>
#include "node_video_loader.h"
//#include "video_loader.h"
#ifdef RALI_VIDEO
//#include "video_loader_module.h"

VideoLoaderNode::VideoLoaderNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<VideoLoaderSharded>(device_resources);
}

/*void VideoLoaderNode::create_node()
{
    std::ostringstream iss;

    // The format for the input string to the OVX decoder API is as follows:
    // <video_stream_count>,<path_to_video_0>:0|1,<path_to_video_0>:0|1,<path_to_video_0>:0|1 ...
    // after each path to videos 0 stands for sw decoding, while 1 stands for hw decoding
    _video_stream_count = _batch_size;
    iss << _video_stream_count << ",";

    auto video_path = _source_path+":";
    std::string path;
    size_t new_pos = 0, prev_pos = 0, vid_count = 0;
    _path_to_videos.resize(_batch_size);
    while((new_pos = video_path.find(":", prev_pos)) != std::string::npos)
    {
        path = video_path.substr(prev_pos, new_pos);
        if(vid_count >= _batch_size)
            break;
        prev_pos = new_pos;
        _path_to_videos[vid_count++] = path;
    }
    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded , using "+((_decode_mode == DecodeMode::USE_HW) ? " hardware decoder ":" software decoder"))
    for(size_t i = 0; i < _video_stream_count; i++)
        iss << (_path_to_videos[i])<< ":" << ((_decode_mode == DecodeMode::USE_HW) ? "1":"0");

    LOG("Total of "+TOSTR(vid_count)+ " videos going to be loaded " + iss.str())
    _interm_output = std::make_unique<Image>(_outputs[0]->info());
    _interm_output->create(_outputs[0]->context());
    _node = amdMediaDecoderNode(_graph->get(), iss.str().c_str(), _interm_output->handle(), NULL);
    _copy_node = vxExtrppNode_Copy(_graph->get(), _interm_output->handle(), _outputs[0]->handle());
    vx_status res = VX_SUCCESS;
    if((res = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Failed to add video decoder node")
    if((res = vxGetStatus((vx_reference)_copy_node)) != VX_SUCCESS)
        THROW("Failed to add video copy decoder node")
}*/

void VideoLoaderNode::init(unsigned internal_shard_count, const std::string &source_path, const std::string &json_path, const std::map<std::string, std::string> feature_key_map, StorageType storage_type,
VideoDecoderType decoder_type, unsigned sequence_length, bool shuffle, bool loop, size_t load_batch_count, RaliMemType mem_type)
{
    //_decode_mode = decoder_mode;
    _source_path = source_path;
    _loop = loop;
    if(!_loader_module)
        THROW("ERROR: loader module is not set for VideoLoaderNode, cannot initialize")
    if(internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the VideoLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, feature_key_map, shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_sequence_length(sequence_length);
    _loader_module->initialize(reader_cfg, VideoDecoderConfig(decoder_type),
             mem_type,
             _batch_size);
    _loader_module->start_loading();
}

/*VideoLoaderNode::VideoLoaderNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs)
{
    _batch_size = outputs[0]->info().batch_size();
    if(_batch_size > MAXIMUM_VIDEO_CONCURRENT_DECODE)
        THROW("Video batch size " + TOSTR(_batch_size)+" is bigger than " + TOSTR(MAXIMUM_VIDEO_CONCURRENT_DECODE))
}*/

std::shared_ptr<VideoLoaderModule> VideoLoaderNode::get_loader_module()
{
    if(!_loader_module)
        WRN("VideoLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

VideoLoaderNode::~VideoLoaderNode()
{
    _loader_module = nullptr;
}

#endif
