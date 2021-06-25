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
#ifdef RALI_VIDEO
extern "C"
{
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}
#include "node_video_loader.h"
#include "node_video_loader_single_shard.h"
#include "video_properties.h"
#endif
#include "rali_api.h"
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
evaluate_image_data_set(RaliImageSizeEvaluationPolicy decode_size_policy, StorageType storage_type,
                        DecoderType decoder_type, const std::string &source_path, const std::string &json_path)
{
    auto translate_image_size_policy = [](RaliImageSizeEvaluationPolicy decode_size_policy)
    {
        switch(decode_size_policy)
        {
            case RALI_USE_MAX_SIZE:
            case RALI_USE_MAX_SIZE_RESTRICTED:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
            case RALI_USE_MOST_FREQUENT_SIZE:
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

auto convert_color_format = [](RaliImageColor color_format)
{
    switch(color_format){
        case RALI_COLOR_RGB24:
            return std::make_tuple(RaliColorFormat::RGB24, 3);

        case RALI_COLOR_BGR24:
            return std::make_tuple(RaliColorFormat::BGR24, 3);

        case RALI_COLOR_U8:
            return std::make_tuple(RaliColorFormat::U8, 1);

        case RALI_COLOR_RGB_PLANAR:
            return std::make_tuple(RaliColorFormat::RGB_PLANAR, 3);

        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_decoder_mode= [](RaliDecodeDevice decode_mode)
{
    switch(decode_mode){
        case RALI_HW_DECODE:
            return DecodeMode::USE_HW;

        case RALI_SW_DECODE:
            return DecodeMode::USE_SW;
        default:

        THROW("Unsupported decoder mode" + TOSTR(decode_mode))
    }
};

RaliImage  RALI_API_CALL
raliJpegFileSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 0, 0, 0,
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegFileSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RaliDecoderType dec_type)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG; // default
        if (dec_type == RALI_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;

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

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "", 0, 0, 0,
                                                                          std::map<std::string, std::string>(),
                                                                          StorageType::FILE_SYSTEM,
                                                                          decType,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliSequenceReader(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_internal_batch_size(context->master_graph->internal_batch_size());
        context->set_user_batch_size(context->master_graph->user_batch_size());
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::SEQUENCE_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->master_graph->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "", sequence_length,
                                                                          step, stride,
                                                                          std::map<std::string, std::string>(),
                                                                          StorageType::SEQUENCE_FILE_SYSTEM,
                                                                          DecoderType::TURBO_JPEG,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliSequenceReaderSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_internal_batch_size(context->master_graph->internal_batch_size());
        context->set_user_batch_size(context->master_graph->user_batch_size());
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::SEQUENCE_FILE_SYSTEM, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 
                                                                                        sequence_length,
                                                                                        step, stride,
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCaffe2LMDBRecordSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "", 0, 0, 0,
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE2_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCaffe2LMDBRecordSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 0, 0, 0,
                                                                                        StorageType::CAFFE2_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCaffeLMDBRecordSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "", 0, 0, 0,
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(), decoder_keep_original);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCaffeLMDBRecordSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 0, 0, 0,
                                                                                        StorageType::CAFFE_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCOCOFileSource(
        RaliContext p_context,
        const char* source_path,
	    const char* json_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                            source_path, json_path, 0, 0, 0,
                                                                            std::map<std::string, std::string>(),
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(), decoder_keep_original);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCOCOFileSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
	const char* json_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG,
                                                       source_path, json_path);
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, json_path, 0, 0, 0,
                                                                                        StorageType::COCO_FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(), decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliFusedJpegCrop(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RaliFloatParam p_area_factor,
        RaliFloatParam p_aspect_ratio,
        RaliFloatParam p_x_drift_factor,
        RaliFloatParam p_y_drift_factor
        )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) ;

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
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::FUSED_TURBO_JPEG,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegCOCOFileSourcePartial(
        RaliContext p_context,
        const char* source_path,
        const char* json_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RaliFloatParam p_area_factor,
        RaliFloatParam p_aspect_ratio,
        RaliFloatParam p_x_drift_factor,
        RaliFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        //bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
                                                                            source_path, json_path,
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(), area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RaliImage  RALI_API_CALL
raliJpegCOCOFileSourcePartialSingleShard(
        RaliContext p_context,
        const char* source_path,
        const char* json_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RaliFloatParam p_area_factor,
        RaliFloatParam p_aspect_ratio,
        RaliFloatParam p_x_drift_factor,
        RaliFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) || (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED);
        //bool decoder_keep_original = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == RALI_USE_MAX_SIZE_RESTRICTED);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                            source_path, json_path,
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(), area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegTFRecordSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        const char* user_key_for_encoded,
        const char* user_key_for_filename,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        std::string user_key_for_encoded_str(user_key_for_encoded);
        std::string user_key_for_filename_str(user_key_for_filename);

        std::map<std::string, std::string> feature_key_map = {
            {"image/encoded",user_key_for_encoded_str},
            {"image/filename",user_key_for_filename_str},
        };


        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "", 0, 0, 0,
                                                                             feature_key_map,
                                                                             StorageType::TF_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliJpegTFRecordSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 0, 0, 0,
                                                                                        StorageType::TF_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliRawTFRecordSource(
        RaliContext p_context,
        const char* source_path,
        const char* user_key_for_encoded_str,
        const char* user_key_for_filename_str,
        RaliImageColor rali_color_format,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned out_width,
        unsigned out_height,
        const char* record_name_prefix)
{
    Image* output = nullptr;
    if(!p_context)
        THROW("Invalid context passed as input")

    auto context = static_cast<Context*>(p_context);
    try
    {
        unsigned internal_shard_count = 1;
        std::map<std::string, std::string> feature_key_map = {
                {"image/encoded",user_key_for_encoded_str},
                {"image/filename",user_key_for_filename_str},
        };

        if(out_width == 0 || out_height == 0)
        {
            THROW("Invalid output width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto info = ImageInfo(out_width, out_height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "", 0, 0, 0,
                                                                             feature_key_map,
                                                                             StorageType::TF_RECORD,
                                                                             DecoderType::SKIP_DECODE,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(), false, record_name_prefix);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliRawTFRecordSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned out_width,
        unsigned out_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if((out_width == 0 || out_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(out_width)+ " height = "+ TOSTR(out_height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(out_width, out_height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "", 0, 0, 0,
                                                                                        StorageType::TF_RECORD,
                                                                                        DecoderType::SKIP_DECODE,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliFusedJpegCropSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RaliFloatParam p_area_factor,
        RaliFloatParam p_aspect_ratio,
        RaliFloatParam p_x_drift_factor,
        RaliFloatParam p_y_drift_factor
        )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE) ;

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
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                          source_path, "",
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::FUSED_TURBO_JPEG,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RaliImage  RALI_API_CALL
raliVideoFileSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        RaliDecodeDevice rali_decode_device,
        unsigned internal_shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        bool shuffle,
        bool is_output,
        bool loop,
        bool file_list_frame_num)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef RALI_VIDEO
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_video_loader_flag();
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_internal_batch_size(context->master_graph->internal_batch_size());
        context->set_user_batch_size(context->master_graph->user_batch_size());
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))

        unsigned width , height, number_of_video_files, frame_rate;
        std::vector<size_t> frames_count;
        std::vector<std::string> video_file_names;
        std::vector<std::tuple<int, int>> start_end_frame_num;

        video_properties video_prop = find_video_properties(source_path, file_list_frame_num);
        width = video_prop.width;
        height = video_prop.height;
        number_of_video_files = video_prop.videos_count;
        frames_count.resize(number_of_video_files);
        video_file_names.resize(number_of_video_files);
        start_end_frame_num.resize(number_of_video_files);
        frames_count = video_prop.frames_count;
        frame_rate = video_prop.frame_rate;
        video_file_names = video_prop.video_file_names;
        start_end_frame_num = video_prop.start_end_frame_num;

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto decoder_mode = convert_decoder_mode(rali_decode_device);
        auto info = ImageInfo(width, height,
                              context->master_graph->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
									                                      std::map<std::string, std::string>(),
                                                                          StorageType::VIDEO_FILE_SYSTEM,
                                                                          VideoDecoderType::FFMPEG_VIDEO,
                                                                          decoder_mode,
                                                                          sequence_length,
                                                                          step,
                                                                          stride,
                                                                          number_of_video_files,
                                                                          frames_count,
                                                                          frame_rate,
                                                                          start_end_frame_num,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          video_file_names);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;

}

RaliImage  RALI_API_CALL
raliVideoFileSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        RaliDecodeDevice rali_decode_device,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        bool shuffle,
        bool is_output,
        bool loop,
        bool file_list_frame_num)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef RALI_VIDEO
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_video_loader_flag();
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_user_batch_size(sequence_length * context->user_batch_size());
        context->set_internal_batch_size(sequence_length);
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))
        
        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")
        
        unsigned width , height, number_of_video_files, frame_rate;
        std::vector<size_t> frames_count;
        std::vector<std::string> video_file_names;
        std::vector<std::tuple<int, int>> start_end_frame_num;

        video_properties video_prop = find_video_properties(source_path, file_list_frame_num);
        width = video_prop.width;
        height = video_prop.height;
        number_of_video_files = video_prop.videos_count;
        frames_count.resize(number_of_video_files);
        video_file_names.resize(number_of_video_files);
        start_end_frame_num.resize(number_of_video_files);
        frames_count = video_prop.frames_count;
        frame_rate = video_prop.frame_rate;
        video_file_names = video_prop.video_file_names;
        start_end_frame_num = video_prop.start_end_frame_num;

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto decoder_mode = convert_decoder_mode(rali_decode_device);
        auto info = ImageInfo(width, height,
                              context->master_graph->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                          source_path, "",
									                                      std::map<std::string, std::string>(),
                                                                          StorageType::VIDEO_FILE_SYSTEM,
                                                                          VideoDecoderType::FFMPEG_VIDEO,
                                                                          decoder_mode,
                                                                          sequence_length,
                                                                          step,
                                                                          stride,
                                                                          number_of_video_files,
                                                                          frames_count,
                                                                          frame_rate,
                                                                          start_end_frame_num,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          video_file_names);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;

}

RaliImage  RALI_API_CALL
raliVideoFileResize(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        RaliDecodeDevice rali_decode_device,
        unsigned internal_shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        unsigned dest_width,
        unsigned dest_height,
        bool shuffle,
        bool is_output,
        bool loop,
        bool file_list_frame_num)
{
    Image* resize_output = nullptr;
    if(!p_context || dest_width == 0 || dest_height == 0)
        THROW("Null values passed as input")

    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef RALI_VIDEO
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_video_loader_flag();
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_internal_batch_size(context->master_graph->internal_batch_size());
        context->set_user_batch_size(context->master_graph->user_batch_size());
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))

        unsigned width , height, number_of_video_files, frame_rate;
        std::vector<size_t> frames_count;
        std::vector<std::string> video_file_names;
        std::vector<std::tuple<int, int>> start_end_frame_num;

        video_properties video_prop = find_video_properties(source_path, file_list_frame_num);
        width = video_prop.width;
        height = video_prop.height;
        number_of_video_files = video_prop.videos_count;
        frames_count.resize(number_of_video_files);
        video_file_names.resize(number_of_video_files);
        start_end_frame_num.resize(number_of_video_files);
        frames_count = video_prop.frames_count;
        frame_rate = video_prop.frame_rate;
        video_file_names = video_prop.video_file_names;
        start_end_frame_num = video_prop.start_end_frame_num;
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto decoder_mode = convert_decoder_mode(rali_decode_device);
        auto info = ImageInfo(width, height,
                              context->master_graph->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        // For the resize node, user can create an image with a different width and height
        ImageInfo output_info = info;
        output_info.width(dest_width);
        output_info.height(dest_height);

        resize_output = context->master_graph->create_image(output_info, false);

        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
									                                      std::map<std::string, std::string>(),
                                                                          StorageType::VIDEO_FILE_SYSTEM,
                                                                          VideoDecoderType::FFMPEG_VIDEO,
                                                                          decoder_mode,
                                                                          sequence_length,
                                                                          step,
                                                                          stride,
                                                                          number_of_video_files,
                                                                          frames_count,
                                                                          frame_rate,
                                                                          start_end_frame_num,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          video_file_names);
        context->master_graph->set_loop(loop);

        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        resize_output->reset_image_roi();

        std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({output}, {resize_output});
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(output_info, is_output);
            context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

RaliImage  RALI_API_CALL
raliVideoFileResizeSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        RaliDecodeDevice rali_decode_device,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        unsigned step,
        unsigned stride,
        unsigned dest_width,
        unsigned dest_height,
        bool shuffle,
        bool is_output,
        bool loop,
        bool file_list_frame_num)
{
    Image* resize_output = nullptr;
    if(!p_context || dest_width == 0 || dest_height == 0)
        THROW("Null values passed as input")

    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef RALI_VIDEO
        /* The internal batch size and user batch size are modified here in master graph */
        context->master_graph->set_video_loader_flag();
        context->master_graph->set_user_internal_batch_size(sequence_length);
        context->master_graph->set_user_batch_size(sequence_length * context->user_batch_size());
        context->master_graph->set_user_internal_batch_ratio();
        context->set_user_batch_size(sequence_length * context->user_batch_size());
        context->set_internal_batch_size(sequence_length);
        INFO("Internal batch size has been set to "+ TOSTR(context->master_graph->internal_batch_size()))

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        unsigned width , height, number_of_video_files, frame_rate;
        std::vector<size_t> frames_count;
        std::vector<std::string> video_file_names;
        std::vector<std::tuple<int, int>> start_end_frame_num;

        video_properties video_prop = find_video_properties(source_path, file_list_frame_num);
        width = video_prop.width;
        height = video_prop.height;
        number_of_video_files = video_prop.videos_count;
        frames_count.resize(number_of_video_files);
        video_file_names.resize(number_of_video_files);
        start_end_frame_num.resize(number_of_video_files);
        frames_count = video_prop.frames_count;
        frame_rate = video_prop.frame_rate;
        video_file_names = video_prop.video_file_names;
        start_end_frame_num = video_prop.start_end_frame_num;
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto decoder_mode = convert_decoder_mode(rali_decode_device);
        auto info = ImageInfo(width, height,
                              context->master_graph->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        // For the resize node, user can create an image with a different width and height
        ImageInfo output_info = info;
        output_info.width(dest_width);
        output_info.height(dest_height);

        resize_output = context->master_graph->create_image(output_info, false);

        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                          source_path, "",
									                                      std::map<std::string, std::string>(),
                                                                          StorageType::VIDEO_FILE_SYSTEM,
                                                                          VideoDecoderType::FFMPEG_VIDEO,
                                                                          decoder_mode,
                                                                          sequence_length,
                                                                          step,
                                                                          stride,
                                                                          number_of_video_files,
                                                                          frames_count,
                                                                          frame_rate,
                                                                          start_end_frame_num,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          video_file_names);
        context->master_graph->set_loop(loop);

        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        resize_output->reset_image_roi();

        std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({output}, {resize_output});
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(output_info, is_output);
            context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

// loader for CFAR10 raw data: Can be used for other raw data loaders as well
RaliImage  RALI_API_CALL
raliRawCIFAR10Source(
                 RaliContext p_context,
                 const char* source_path,
                 RaliImageColor rali_color_format,
                 bool is_output ,
                 unsigned out_width,
                 unsigned out_height,
                 const char* filename_prefix,
                 bool loop )
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(out_width == 0 || out_height == 0)
        {
            THROW("Invalid video input width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height));
        }

        auto [width, height] = std::make_tuple(out_width, out_height);
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<Cifar10LoaderNode>({}, {output})->init(source_path, "",
                                                                             StorageType::UNCOMPRESSED_BINARY_DATA,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(), filename_prefix);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RaliStatus RALI_API_CALL
raliResetLoaders(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->reset();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}
