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

#pragma once
#include <vector>
#include "audio_loader.h"
//
// AudioLoaderSharded Can be used to run load and decode in multiple shards, each shard by a single loader instance,
// It improves load and decode performance since each loader loads the audios in parallel using an internal thread
//
class AudioLoaderSharded : public LoaderModule
{
public:
#if ENABLE_HIP
    explicit AudioLoaderSharded(DeviceResourcesHip dev_resources);
#else
    explicit AudioLoaderSharded(DeviceResources dev_resources);
#endif
    ~AudioLoaderSharded() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size=false) override;
    void set_output (rocALTensor* output_audio) override;
    // void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    decoded_image_info get_decode_image_info() override;
    crop_image_info get_crop_image_info() override;
    Timing timing() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
private:
    void increment_loader_idx();
#if ENABLE_HIP
    const DeviceResourcesHip _dev_resources;
#else
    const DeviceResources _dev_resources;
#endif
    bool _initialized = false;
    std::vector<std::shared_ptr<AudioLoader>> _loaders;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();
    size_t _prefetch_queue_depth;

    rocALTensor *_output_tensor;
    bool tensor = true; // for debugging purpose should be removed
    // std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
};