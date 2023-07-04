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

#include "commons.h"
#include "context.h"
#include "rocal_api.h"

size_t  ROCAL_API_CALL
rocalGetRemainingImages(RocalContext p_context)
{
    auto context = static_cast<Context *>(p_context);
    size_t count = 0;
    try
    {
        count = context->master_graph->remaining_count();
    }
    catch (const std::exception &e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

size_t  ROCAL_API_CALL
rocalGetLastBatchPaddedSize(RocalContext p_context)
{
    auto context = static_cast<Context *>(p_context);
    size_t count = 0;
    try
    {
        count = context->master_graph->last_batch_size();
    }
    catch (const std::exception &e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

RocalStatus ROCAL_API_CALL rocalGetStatus(RocalContext p_context)
{
    if (!p_context)
        return ROCAL_CONTEXT_INVALID;
    auto context = static_cast<Context *>(p_context);

    if (context->no_error())
        return ROCAL_OK;

    return ROCAL_RUNTIME_ERROR;
}

const char *ROCAL_API_CALL rocalGetErrorMessage(RocalContext p_context)
{
    auto context = static_cast<Context *>(p_context);
    return context->error_msg();
}
TimingInfo
ROCAL_API_CALL rocalGetTimingInfo(RocalContext p_context)
{
    auto context = static_cast<Context *>(p_context);
    auto info = context->timing();
    // INFO("bbencode time "+ TOSTR(info.bb_process_time)); //to display time taken for bbox encoder

    return {info.image_read_time, info.image_decode_time, info.image_process_time, info.copy_to_output, info.wait_if_empty_time, info.wait_if_full_time, info.circular_buffer_wait_if_empty_time, info.circular_buffer_wait_if_full_time};
}

RocalMetaData
    ROCAL_API_CALL
    rocalCreateCaffe2LMDBLabelReader(RocalContext p_context, const char *source_path, bool is_output)
{

    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateCaffe2LMDBLabelReader")

    auto context = static_cast<Context *>(p_context);
    return context->master_graph->create_caffe2_lmdb_record_meta_data_reader(source_path, MetaDataReaderType::CAFFE2_META_DATA_READER, MetaDataType::Label);
}

RocalMetaData
    ROCAL_API_CALL
    rocalCreateCaffe2LMDBReaderDetection(RocalContext p_context, const char *source_path, bool is_output)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateCaffe2LMDBReaderDetection")
    auto context = static_cast<Context *>(p_context);

    return context->master_graph->create_caffe2_lmdb_record_meta_data_reader(source_path, MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER, MetaDataType::BoundingBox);
}

RocalMetaData
    ROCAL_API_CALL
    rocalCreateCaffeLMDBLabelReader(RocalContext p_context, const char *source_path)
{

    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateCaffeLMDBLabelReader")
    auto context = static_cast<Context *>(p_context);
    return context->master_graph->create_caffe_lmdb_record_meta_data_reader(source_path, MetaDataReaderType::CAFFE_META_DATA_READER, MetaDataType::Label);
}

RocalMetaData
    ROCAL_API_CALL
    rocalCreateCaffeLMDBReaderDetection(RocalContext p_context, const char *source_path)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateCaffeLMDBReaderDetection")
    auto context = static_cast<Context *>(p_context);

    return context->master_graph->create_caffe_lmdb_record_meta_data_reader(source_path, MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER, MetaDataType::BoundingBox);
}

size_t ROCAL_API_CALL rocalIsEmpty(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalIsEmpty")
    auto context = static_cast<Context *>(p_context);
    size_t ret = 0;
    try
    {
        ret = context->master_graph->empty();
    }
    catch (const std::exception &e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return ret;
}
