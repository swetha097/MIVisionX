/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "node_audio_loader_single_shard.h"
#include "exception.h"


AudioLoaderSingleShardNode::AudioLoaderSingleShardNode(rocalTensor *output, void* device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<AudioLoader>(device_resources);
}

void
AudioLoaderSingleShardNode::init(unsigned shard_id, unsigned shard_count, const std::string &source_path, const std::string &source_list_path,
                                 StorageType storage_type, DecoderType decoder_type, bool shuffle, bool loop,
                                 size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader, RocalBatchPolicy last_batch_policy, bool last_batch_padded, bool stick_to_shard, signed shard_size)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for AudioLoaderNode, cannot initialize")
    if(shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    if(shard_id >= shard_count)
        THROW("Shard is should be smaller than shard count")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the AudioLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, source_list_path, std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(shard_count);
    reader_cfg.set_shard_id(shard_id);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_meta_data_reader(meta_data_reader);
    reader_cfg.set_last_batch_policy(last_batch_policy, last_batch_padded);
    reader_cfg.set_stick_to_shard(stick_to_shard);
    reader_cfg.set_shard_size(shard_size);
    _loader_module->initialize(reader_cfg, DecoderConfig(decoder_type),
                               mem_type,
                               _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> AudioLoaderSingleShardNode::get_loader_module()
{
    if(!_loader_module)
        WRN("AudioLoaderSingleShardNode's loader module is null, not initialized")
    return _loader_module;
}

AudioLoaderSingleShardNode::~AudioLoaderSingleShardNode()
{
    _loader_module = nullptr;
}
