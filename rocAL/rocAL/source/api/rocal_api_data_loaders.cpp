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

#include <tuple>
#include <assert.h>
#include <boost/filesystem.hpp>
#ifdef ROCAL_VIDEO
#include "node_video_loader.h"
#include "node_video_loader_single_shard.h"
#endif
#include "rocal_api.h"
#include "commons.h"
#include "context.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "node_cifar10_loader.h"
#include "image_source_evaluator.h"
#include "node_fisheye.h"
#include "node_copy.h"
#include "node_fused_jpeg_crop.h"
#include "node_fused_jpeg_crop_single_shard.h"
#include "node_resize.h"
#include "meta_node_resize.h"

namespace filesys = boost::filesystem;

std::tuple<unsigned, unsigned>
evaluate_image_data_set(RocalImageSizeEvaluationPolicy decode_size_policy, StorageType storage_type,
                        DecoderType decoder_type, const std::string &source_path, const std::string &json_path)
{
    auto translate_image_size_policy = [](RocalImageSizeEvaluationPolicy decode_size_policy)
    {
        switch(decode_size_policy)
        {
            case ROCAL_USE_MAX_SIZE:
            case ROCAL_USE_MAX_SIZE_RESTRICTED:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
            case ROCAL_USE_MOST_FREQUENT_SIZE:
                return MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
            default:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
        }
    };

    ImageSourceEvaluator source_evaluator;
    source_evaluator.set_size_evaluation_policy(translate_image_size_policy(decode_size_policy));
    if(source_evaluator.create(ReaderConfig(storage_type, source_path, json_path), DecoderConfig(decoder_type)) != ImageSourceEvaluatorStatus::OK)
        THROW("Initializing file source input evaluator failed ")
    auto max_width = source_evaluator.max_width();
    auto max_height = source_evaluator.max_height();
    if(max_width == 0 ||max_height  == 0)
        THROW("Cannot find size of the images or images cannot be accessed")

    LOG("Maximum input image dimension [ "+ TOSTR(max_width) + " x " + TOSTR(max_height)+" ] for images in "+source_path)
    return std::make_tuple(max_width, max_height);
};

auto convert_color_format = [](RocalImageColor color_format)
{
    switch(color_format){
        case ROCAL_COLOR_RGB24:
            return std::make_tuple(RocalColorFormat::RGB24, 3);

        case ROCAL_COLOR_BGR24:
            return std::make_tuple(RocalColorFormat::BGR24, 3);

        case ROCAL_COLOR_U8:
            return std::make_tuple(RocalColorFormat::U8, 1);

        case ROCAL_COLOR_RGB_PLANAR:
            return std::make_tuple(RocalColorFormat::RGB_PLANAR, 3);

        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_decoder_mode= [](RocalDecodeDevice decode_mode)
{
    switch(decode_mode){
        case ROCAL_HW_DECODE:
            return DecodeMode::HW_VAAPI;

        case ROCAL_SW_DECODE:
            return DecodeMode::CPU;
        default:

            THROW("Unsupported decoder mode" + TOSTR(decode_mode))
    }
};

RocalTensor  ROCAL_API_CALL
rocalJpegFileSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        RocalTensorFormat tensor_format = RocalTensorFormat::NHWC;
        RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
        auto info  = TensorInfo(width,
                                height,
                                context->internal_batch_size(),
                                num_of_planes,
                                context->master_graph->mem_type(),
                                color_format,
                                tensor_data_type,
                                tensor_format);
        output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original
                                                                                        );
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_tensor_node<CopyTensorNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RocalTensor  ROCAL_API_CALL
rocalJpegFileSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalDecoderType dec_type)
{
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG; // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        RocalTensorFormat tensor_format = RocalTensorFormat::NHWC;
        RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
        auto info  = TensorInfo(width,
                                height,
                                context->internal_batch_size(),
                                num_of_planes,
                                context->master_graph->mem_type(),
                                color_format,
                                tensor_data_type,
                                tensor_format);
        output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_tensor_node<ImageLoaderTensorNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
                                                                          std::map<std::string, std::string>(),
                                                                          StorageType::FILE_SYSTEM,
                                                                          decType,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          context->master_graph->meta_data_reader(),
                                                                          decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_tensor_node<CopyTensorNode>({output}, {actual_output}); // Have to add copy tensor node
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}
