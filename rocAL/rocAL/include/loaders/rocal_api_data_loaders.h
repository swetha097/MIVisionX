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

#ifndef MIVISIONX_ROCAL_API_DATA_LOADERS_H
#define MIVISIONX_ROCAL_API_DATA_LOADERS_H
#include "../api/rocal_api_types.h"

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image

extern "C"  RaliTensor  ROCAL_API_CALL rocalJpegFileSource(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor rali_color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0, RaliDecoderType rali_decoder_type=RaliDecoderType::ROCAL_DECODER_TJPEG);


/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image

extern "C"  RaliTensor  ROCAL_API_CALL rocalJpegFileSourceSingleShard(RaliContext context,
                                                                   const char* source_path,
                                                                   RaliImageColor rali_color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   RaliImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0);
