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

//
// Created by mvx on 3/31/20.
//

#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#define MAX_BUFFER 10000

void
ROCAL_API_CALL rocalRandomBBoxCrop(RocalContext p_context, bool all_boxes_overlap, bool no_crop, RocalFloatParam p_aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, RocalFloatParam p_scaling, int total_num_attempts, int64_t seed)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalRandomBBoxCrop")
    auto context = static_cast<Context*>(p_context);
    FloatParam *aspect_ratio;
    FloatParam *scaling;
    if(p_aspect_ratio == NULL)
    {
        aspect_ratio = ParameterFactory::instance()->create_uniform_float_rand_param(1.0, 1.0);
    }
    else
    {

        aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    }
    if(p_scaling == NULL)
    {
        scaling = ParameterFactory::instance()->create_uniform_float_rand_param(1.0, 1.0);
    }
    else
    {
        scaling = static_cast<FloatParam*>(p_scaling);
    }
    context->master_graph->create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType::RandomBBoxCropReader, RandomBBoxCrop_MetaDataType::BoundingBox, all_boxes_overlap, no_crop, aspect_ratio, has_shape, crop_width, crop_height, num_attempts, scaling, total_num_attempts, seed);
}

RocalMetaData
ROCAL_API_CALL rocalCreateLabelReader(RocalContext p_context, const char* source_path) {
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateLabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::FOLDER_BASED_LABEL_READER);
}

RocalMetaData
ROCAL_API_CALL rocalCreateCOCOReader(RocalContext p_context, const char* source_path, bool is_output) {
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateCOCOReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_coco_meta_data_reader(source_path, is_output, MetaDataReaderType::COCO_META_DATA_READER,  MetaDataType::BoundingBox);
}

void
ROCAL_API_CALL rocalGetImageName(RocalContext p_context,  char* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageName")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data_info();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        memcpy(buf, meta_data.first[i].c_str(), meta_data.first[i].size());
        buf += meta_data.first[i].size() * sizeof(char);
    }
}

unsigned
ROCAL_API_CALL rocalGetImageNameLen(RocalContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageNameLen")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data_info();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        buf[i] = meta_data.first[i].size();
        size += buf[i];
    }
    return size;
}

RocalTensorList
ROCAL_API_CALL rocalGetImageLabels(RocalContext p_context)
{

    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageLabels")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->labels_meta_data();
}

RocalTensorList
ROCAL_API_CALL rocalGetBoundingBoxLabel(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetBoundingBoxLabel")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->bbox_labels_meta_data();
}

RocalTensorList
ROCAL_API_CALL rocalGetBoundingBoxCords(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetBoundingBoxCords")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->bbox_meta_data();
}

#if 0 // Commented out for now
void
ROCAL_API_CALL rocalGetOneHotImageLabels(RocalContext p_context, int* buf, int numOfClasses)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetOneHotImageLabels")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second) {
        WRN("No label has been loaded for this output image")
        return;
    }
    size_t meta_data_batch_size = meta_data.second->get_label_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))

    int labels_buf[meta_data_batch_size];
    int one_hot_encoded[meta_data_batch_size*numOfClasses];
    memset(one_hot_encoded, 0, sizeof(int) * meta_data_batch_size * numOfClasses);
    memcpy(labels_buf, meta_data.second->get_label_batch().data(),  sizeof(int)*meta_data_batch_size);

    for(uint i = 0; i < meta_data_batch_size; i++)
    {
        int label_index =  labels_buf[i];
        if (label_index >0 && label_index<= numOfClasses )
        {
        one_hot_encoded[(i*numOfClasses)+label_index-1]=1;

        }
        else if(label_index == 0)
        {
          one_hot_encoded[(i*numOfClasses)+numOfClasses-1]=1;
        }

    }
    memcpy(buf,one_hot_encoded, sizeof(int) * meta_data_batch_size * numOfClasses);
}

void
ROCAL_API_CALL rocalGetImageSizes(RocalContext p_context, int* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageSizes")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_img_sizes_batch().size();


    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        memcpy(buf, &(meta_data.second->get_img_sizes_batch()[i]), sizeof(ImgSize));
        buf += 2;
    }
}
#endif
void ROCAL_API_CALL rocalBoxEncoder(RocalContext p_context, std::vector<float>& anchors, float criteria,
                                  std::vector<float> &means, std::vector<float> &stds, bool offset, float scale)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalBoxEncoder")
    auto context = static_cast<Context *>(p_context);
    context->master_graph->box_encoder(anchors, criteria, means, stds, offset, scale);
}

void
ROCAL_API_CALL rocalCopyEncodedBoxesAndLables(RocalContext p_context, float* boxes_buf, int* labels_buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCopyEncodedBoxesAndLables")
    auto context = static_cast<Context *>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if (context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != " + TOSTR(context->user_batch_size()))
    if (!meta_data.second)
    {
        WRN("No encoded labels and bounding boxes has been loaded for this output image")
        return;
    }
    unsigned sum = 0;
    unsigned bb_offset[meta_data_batch_size];
    for (unsigned i = 0; i < meta_data_batch_size; i++)
    {
        bb_offset[i] = sum;
        sum += meta_data.second->get_bb_labels_batch()[i].size();
    }
    // copy labels buffer & bboxes buffer parallely
    #pragma omp parallel for
    for (unsigned i = 0; i < meta_data_batch_size; i++)
    {
        unsigned bb_count = meta_data.second->get_bb_labels_batch()[i].size();
        int *temp_labels_buf = labels_buf + bb_offset[i];
        float *temp_bbox_buf = boxes_buf + (bb_offset[i] * 4);
        memcpy(temp_labels_buf, meta_data.second->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(temp_bbox_buf, meta_data.second->get_bb_cords_batch()[i].data(), sizeof(BoundingBoxCord) * bb_count);
    }
}

void
ROCAL_API_CALL rocalGetEncodedBoxesAndLables(RocalContext p_context, float **boxes_buf_ptr, int **labels_buf_ptr, int num_encoded_boxes)
{
    if (!p_context) {
        WRN("rocalGetEncodedBoxesAndLables::Invalid context")
    }
    auto context = static_cast<Context *>(p_context);
    context->master_graph->get_bbox_encoded_buffers(boxes_buf_ptr, labels_buf_ptr, num_encoded_boxes);
    if (!*boxes_buf_ptr || !*labels_buf_ptr)
    {
        WRN("rocalGetEncodedBoxesAndLables::Empty tensors returned from rocAL")
    }
}
