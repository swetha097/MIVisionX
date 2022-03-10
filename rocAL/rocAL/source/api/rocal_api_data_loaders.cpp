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

#include <tuple>
#include "rocal_api.h"
#include "commons.h"
#include "context.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "image_source_evaluator.h"
#include "node_copy.h"
#include "node_fused_jpeg_crop.h"
#include "node_fused_jpeg_crop_single_shard.h"

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

//********************************************************************************************************************

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
    rocALTensor* output = nullptr;
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

        RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
        RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
        RocalROIType roi_type = RocalROIType::XYWH;
        unsigned num_of_dims = 4;
        std::vector<unsigned> dims;
        dims.resize(num_of_dims);
        dims.at(0) = context->internal_batch_size();
        dims.at(1) = height;
        dims.at(2) = width;
        dims.at(3) = num_of_planes;
        auto info  = rocALTensorInfo(num_of_dims,
                                std::make_shared<std::vector<unsigned> >(std::move(dims)),
                                context->master_graph->mem_type(),
                                tensor_data_type);
        info.set_roi_type(roi_type);
        info.set_color_format(color_format);
        info.set_tensor_layout(tensor_format);
        output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
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
            // context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
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
    rocALTensor* output = nullptr;
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

        RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
        RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
        RocalROIType roi_type = RocalROIType::XYWH;
        unsigned num_of_dims = 4;
        std::vector<unsigned> dims;
        dims.resize(4);
        dims[0] = context->internal_batch_size();
        dims[1] = height;
        dims[2] = width;
        dims[3] = num_of_planes;
        std::cerr<<"\n dims"<<dims[0]<<dims[1]<<dims[2]<<dims[3];
        auto info  = rocALTensorInfo(num_of_dims,
                                std::make_shared<std::vector<unsigned> >(std::move(dims)),
                                context->master_graph->mem_type(),
                                tensor_data_type);
        info.set_roi_type(roi_type);
        info.set_color_format(color_format);
        info.set_tensor_layout(tensor_format);
        output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
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
            context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output}); // Have to add copy tensor node
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
rocalAudioFileSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_frames,
        unsigned max_channels)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        // Audio tensor length is dependent on the longest audio sample present in a batch so following variables are not needed (to be removed)
        // bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        // bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // if(use_input_dimension && (max_width == 0 || max_height == 0))
        // {
        //     THROW("Invalid input max width and height");
        // }
        // else
        // {
        //     LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        // }

        // TODO: Add evaluate_audio_data_set function to get the largest audio sample length in a batch
        auto [max_frames, max_channels] = evaluate_audio_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::SNDFILE,
                                                       source_path, "");

        INFO("Internal buffer size for audio frames = "+ TOSTR(max_frames))

        RocalTensorlayout tensor_format = RocalTensorlayout::NONE;
        RocalTensorDataType tensor_data_type = RocalTensorDataType::FP32;
        RocalROIType roi_type = RocalROIType::XYWH;  // Letting the roi_type be default value since it isn't required for audio decoder
        unsigned num_of_dims = 2;
        std::shared_ptr<std::vector<unsigned>> dims;
        dims->resize(num_of_dims);
        dims->at(0) = context->internal_batch_size();
        dims->at(1) = max_frames;
        dims->at(2) = max_channels;
        auto info  = rocALTensorInfo(num_of_dims,
                                dims,
                                context->master_graph->mem_type(),
                                roi_type,
                                tensor_data_type,
                                tensor_format,
                                rocal_color_format); // To be removed since there's no color format for audio frames
        output = context->master_graph->create_loader_output_tensor(info);
        // TODO: Add a loader module for loading audio files from filesystem
        context->master_graph->add_node<AudioLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        DecoderType::SNDFILE,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original  // Can remove this since not needed
                                                                                        );
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            // context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
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
rocalJpegCaffe2LMDBRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
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
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

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

        context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE2_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
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
            context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
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
rocalJpegCaffe2LMDBRecordSourceSingleShard(
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
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
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
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
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
                                                                                        StorageType::CAFFE2_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
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
            context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
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
rocalJpegCaffeLMDBRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
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
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

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

        context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
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
            context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
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
rocalJpegCaffeLMDBRecordSourceSingleShard(
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
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
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
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
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
                                                                                        StorageType::CAFFE_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
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
            context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

//********************************************************************************************************************

// RocalTensor  ROCAL_API_CALL
// rocalJpegFileSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);
//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, "",
//                                                                                         StorageType::FILE_SYSTEM,
//                                                                                         DecoderType::TURBO_JPEG,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                                         context->master_graph->meta_data_reader(),
//                                                                                         decoder_keep_original
//                                                                                         );
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegFileSource(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height,
//         RocalDecoderType dec_type)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
//         DecoderType decType = DecoderType::TURBO_JPEG; // default
//         if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;

//         if(internal_shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);
//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                           source_path, "",
//                                                                           std::map<std::string, std::string>(),
//                                                                           StorageType::FILE_SYSTEM,
//                                                                           decType,
//                                                                           shuffle,
//                                                                           loop,
//                                                                           context->user_batch_size(),
//                                                                           context->master_graph->mem_type(),
//                                                                           context->master_graph->meta_data_reader(),
//                                                                           decoder_keep_original);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output}); // Have to add copy tensor node
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegCaffe2LMDBRecordSource(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(internal_shard_count < 1 )
//             THROW("internal shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                              source_path, "",
//                                                                              std::map<std::string, std::string>(),
//                                                                              StorageType::CAFFE2_LMDB_RECORD,
//                                                                              DecoderType::TURBO_JPEG,
//                                                                              shuffle,
//                                                                              loop,
//                                                                              context->user_batch_size(),
//                                                                              context->master_graph->mem_type(),
//                                                                              context->master_graph->meta_data_reader(),
//                                                                              decoder_keep_original);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalJpegCaffe2LMDBRecordSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, "",
//                                                                                         StorageType::CAFFE2_LMDB_RECORD,
//                                                                                         DecoderType::TURBO_JPEG,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                                         context->master_graph->meta_data_reader(),
//                                                                                         decoder_keep_original);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegCaffeLMDBRecordSource(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(internal_shard_count < 1 )
//             THROW("internal shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                              source_path, "",
//                                                                              std::map<std::string, std::string>(),
//                                                                              StorageType::CAFFE_LMDB_RECORD,
//                                                                              DecoderType::TURBO_JPEG,
//                                                                              shuffle,
//                                                                              loop,
//                                                                              context->user_batch_size(),
//                                                                              context->master_graph->mem_type(),
//                                                                              context->master_graph->meta_data_reader(),
//                                                                              decoder_keep_original);

//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalJpegCaffeLMDBRecordSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, "",
//                                                                                         StorageType::CAFFE_LMDB_RECORD,
//                                                                                         DecoderType::TURBO_JPEG,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                                         context->master_graph->meta_data_reader(),
//                                                                                         decoder_keep_original);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegCOCOFileSource(
//         RocalContext p_context,
//         const char* source_path,
// 	    const char* json_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(internal_shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, json_path);

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                             source_path, json_path,
//                                                                             std::map<std::string, std::string>(),
//                                                                             StorageType::COCO_FILE_SYSTEM,
//                                                                             DecoderType::TURBO_JPEG,
//                                                                             shuffle,
//                                                                             loop,
//                                                                             context->user_batch_size(),
//                                                                             context->master_graph->mem_type(),
//                                                                             context->master_graph->meta_data_reader(),
//                                                                             decoder_keep_original);

//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalJpegCOCOFileSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         const char* json_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG,
//                                                        source_path, json_path);
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, json_path,
//                                                                                         StorageType::COCO_FILE_SYSTEM,
//                                                                                         DecoderType::TURBO_JPEG,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                                         context->master_graph->meta_data_reader(),
//                                                                                         decoder_keep_original);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegTFRecordSource(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         const char* user_key_for_encoded,
//         const char* user_key_for_filename,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         std::string user_key_for_encoded_str(user_key_for_encoded);
//         std::string user_key_for_filename_str(user_key_for_filename);

//         std::map<std::string, std::string> feature_key_map = {
//             {"image/encoded",user_key_for_encoded_str},
//             {"image/filename",user_key_for_filename_str},
//         };


//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);

//         if(internal_shard_count < 1 )
//             THROW("internal shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                              source_path, "",
//                                                                              feature_key_map,
//                                                                              StorageType::TF_RECORD,
//                                                                              DecoderType::TURBO_JPEG,
//                                                                              shuffle,
//                                                                              loop,
//                                                                              context->user_batch_size(),
//                                                                              context->master_graph->mem_type(),
//                                                                              context->master_graph->meta_data_reader());
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalJpegCOCOFileSourcePartial(
//         RocalContext p_context,
//         const char* source_path,
//         const char* json_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height,
//         RocalFloatParam p_area_factor,
//         RocalFloatParam p_aspect_ratio,
//         RocalFloatParam p_x_drift_factor,
//         RocalFloatParam p_y_drift_factor )
// {
//     Tensor* output = nullptr;
//     auto area_factor  = static_cast<FloatParam*>(p_area_factor);
//     auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
//     auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
//     auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         //bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(internal_shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info = TensorInfo(width, height,
//                               context->internal_batch_size(),
//                               num_of_planes,
//                               context->master_graph->mem_type(),
//                               color_format,
//                               tensor_data_type,
//                               tensor_format );
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
//                                                                             source_path, json_path,
//                                                                             StorageType::COCO_FILE_SYSTEM,
//                                                                             DecoderType::FUSED_TURBO_JPEG,
//                                                                             shuffle,
//                                                                             loop,
//                                                                             context->user_batch_size(),
//                                                                             context->master_graph->mem_type(),
//                                                                             context->master_graph->meta_data_reader(),
//                                                                             area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalJpegCOCOFileSourcePartialSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         const char* json_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height,
//         RocalFloatParam p_area_factor,
//         RocalFloatParam p_aspect_ratio,
//         RocalFloatParam p_x_drift_factor,
//         RocalFloatParam p_y_drift_factor )
// {
//     Tensor* output = nullptr;
//     auto area_factor  = static_cast<FloatParam*>(p_area_factor);
//     auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
//     auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
//     auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
//         //bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info = TensorInfo(width, height,
//                               context->internal_batch_size(),
//                               num_of_planes,
//                               context->master_graph->mem_type(),
//                               color_format,
//                               tensor_data_type,
//                               tensor_format );
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<FusedJpegCropTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                             source_path, json_path,
//                                                                             StorageType::COCO_FILE_SYSTEM,
//                                                                             DecoderType::FUSED_TURBO_JPEG,
//                                                                             shuffle,
//                                                                             loop,
//                                                                             context->user_batch_size(),
//                                                                             context->master_graph->mem_type(),
//                                                                             context->master_graph->meta_data_reader(),
//                                                                             area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalJpegTFRecordSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE);

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
//                                                        source_path, "");
//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, "",
//                                                                                         StorageType::TF_RECORD,
//                                                                                         DecoderType::TURBO_JPEG,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                              context->master_graph->meta_data_reader());
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


// RocalTensor  ROCAL_API_CALL
// rocalFusedJpegCrop(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned internal_shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height,
//         RocalFloatParam p_area_factor,
//         RocalFloatParam p_aspect_ratio,
//         RocalFloatParam p_x_drift_factor,
//         RocalFloatParam p_y_drift_factor
//         )
// {
//     Tensor* output = nullptr;
//     auto area_factor  = static_cast<FloatParam*>(p_area_factor);
//     auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
//     auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
//     auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) ;

//         if(internal_shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);
//         context->master_graph->add_tensor_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
//                                                                           source_path, "",
//                                                                           StorageType::FILE_SYSTEM,
//                                                                           DecoderType::FUSED_TURBO_JPEG,
//                                                                           shuffle,
//                                                                           loop,
//                                                                           context->user_batch_size(),
//                                                                           context->master_graph->mem_type(),
//                                                                           context->master_graph->meta_data_reader(),
//                                                                           area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalRawTFRecordSource(
//         RocalContext p_context,
//         const char* source_path,
//         const char* user_key_for_encoded_str,
//         const char* user_key_for_filename_str,
//         RocalImageColor rocal_color_format,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         unsigned out_width,
//         unsigned out_height,
//         const char* record_name_prefix)
// {
//     Tensor* output = nullptr;
//     if(!p_context)
//         THROW("Invalid context passed as input")

//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         unsigned internal_shard_count = 1;
//         std::map<std::string, std::string> feature_key_map = {
//                 {"image/encoded",user_key_for_encoded_str},
//                 {"image/filename",user_key_for_filename_str},
//         };

//         if(out_width == 0 || out_height == 0)
//         {
//             THROW("Invalid output width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
//         }

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info = TensorInfo(out_width, out_height,
//                               context->internal_batch_size(),
//                               num_of_planes,
//                               context->master_graph->mem_type(),
//                               color_format,
//                               tensor_data_type,
//                               tensor_format );
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
//                                                                              source_path, "",
//                                                                              feature_key_map,
//                                                                              StorageType::TF_RECORD,
//                                                                              DecoderType::SKIP_DECODE,
//                                                                              shuffle,
//                                                                              loop,
//                                                                              context->user_batch_size(),
//                                                                              context->master_graph->mem_type(),
//                                                                              context->master_graph->meta_data_reader(),
//                                                                              false, record_name_prefix);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalRawTFRecordSourceSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         unsigned out_width,
//         unsigned out_height)
// {
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     try
//     {

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if((out_width == 0 || out_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
//         }

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(out_width)+ " height = "+ TOSTR(out_height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info = TensorInfo(out_width, out_height,
//                               context->internal_batch_size(),
//                               num_of_planes,
//                               context->master_graph->mem_type(),
//                               color_format,
//                               tensor_data_type,
//                                tensor_format );
//         output = context->master_graph->create_loader_output_tensor(info);

//         context->master_graph->add_tensor_node<ImageLoaderTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                                         source_path, "",
//                                                                                         StorageType::TF_RECORD,
//                                                                                         DecoderType::SKIP_DECODE,
//                                                                                         shuffle,
//                                                                                         loop,
//                                                                                         context->user_batch_size(),
//                                                                                         context->master_graph->mem_type(),
//                                                                                         context->master_graph->meta_data_reader());
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }

// RocalTensor  ROCAL_API_CALL
// rocalFusedJpegCropSingleShard(
//         RocalContext p_context,
//         const char* source_path,
//         RocalImageColor rocal_color_format,
//         unsigned shard_id,
//         unsigned shard_count,
//         bool is_output,
//         bool shuffle,
//         bool loop,
//         RocalImageSizeEvaluationPolicy decode_size_policy,
//         unsigned max_width,
//         unsigned max_height,
//         RocalFloatParam p_area_factor,
//         RocalFloatParam p_aspect_ratio,
//         RocalFloatParam p_x_drift_factor,
//         RocalFloatParam p_y_drift_factor
//         )
// {
//     Tensor* output = nullptr;
//     auto area_factor  = static_cast<FloatParam*>(p_area_factor);
//     auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
//     auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
//     auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
//     auto context = static_cast<Context*>(p_context);
//     try
//     {
//         bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) ;

//         if(shard_count < 1 )
//             THROW("Shard count should be bigger than 0")

//         if(shard_id >= shard_count)
//             THROW("Shard id should be smaller than shard count")

//         if(use_input_dimension && (max_width == 0 || max_height == 0))
//         {
//             THROW("Invalid input max width and height");
//         }
//         else
//         {
//             LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
//         }

//         auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
//                                evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

//         auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

//         INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

//         RocalTensorlayout tensor_format = RocalTensorlayout::NHWC;
//         RocalTensorDataType tensor_data_type = RocalTensorDataType::UINT8;
//         auto info  = TensorInfo(width,
//                                 height,
//                                 context->internal_batch_size(),
//                                 num_of_planes,
//                                 context->master_graph->mem_type(),
//                                 color_format,
//                                 tensor_data_type,
//                                 tensor_format);
//         output = context->master_graph->create_loader_output_tensor(info);
//         context->master_graph->add_tensor_node<FusedJpegCropTensorSingleShardNode>({}, {output})->init(shard_id, shard_count,
//                                                                           source_path, "",
//                                                                           StorageType::FILE_SYSTEM,
//                                                                           DecoderType::FUSED_TURBO_JPEG,
//                                                                           shuffle,
//                                                                           loop,
//                                                                           context->user_batch_size(),
//                                                                           context->master_graph->mem_type(),
//                                                                           context->master_graph->meta_data_reader(),
//                                                                           area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
//         context->master_graph->set_loop(loop);

//         if(is_output)
//         {
//             auto actual_output = context->master_graph->create_tensor(info, is_output);
//             context->master_graph->add_tensor_node<CopyNode>({output}, {actual_output});
//         }

//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         std::cerr << e.what() << '\n';
//     }
//     return output;
// }


RocalStatus ROCAL_API_CALL
rocalResetLoaders(RocalContext p_context)
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
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
