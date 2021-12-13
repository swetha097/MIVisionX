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
#include "node.h"
#include "image_loader_sharded.h"
#include "graph.h"

class ImageLoaderSingleShardNode : public Node
{
public:
#if ENABLE_HIP
    ImageLoaderSingleShardNode(Image *output, DeviceResourcesHip device_resources);
#else
    ImageLoaderSingleShardNode(Image *output, DeviceResources device_resources);
#endif
    ~ImageLoaderSingleShardNode() override;

    /// \param user_shard_count shard count from user
    /// \param  user_shard_id shard id from user
    /// \param source_path Defines the path that includes the image dataset
    /// \param load_batch_count Defines the quantum count of the images to be loaded. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images in multiples of the load_batch_count,
    /// for example if there are 10 images in the dataset and load_batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    void init(unsigned shard_id, unsigned shard_count, const std::string &source_path, const std::string &json_path, StorageType storage_type, DecoderType decoder_type,
              bool shuffle, bool loop, size_t load_batch_count, RaliMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader, bool decoder_keep_orig = false,
              const std::map<std::string, std::string> feature_key_map = std::map<std::string, std::string>(), unsigned sequence_length = 0, unsigned step = 0, unsigned stride = 0);

    std::shared_ptr<LoaderModule> get_loader_module();
protected:
    void create_node() override{};
    void update_node() override{};
private:
    std::shared_ptr<ImageLoader> _loader_module = nullptr;
};