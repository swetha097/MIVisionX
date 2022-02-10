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
#include "rali_api.h"
#define MAX_BUFFER 10000

RaliMetaData
RALI_API_CALL raliCreateLabelReader(RaliContext p_context, const char* source_path) {
    if (!p_context)
        THROW("Invalid rali context passed to raliCreateLabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::FOLDER_BASED_LABEL_READER);

}

void
RALI_API_CALL raliGetImageName(RaliContext p_context,  char* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageName")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
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
RALI_API_CALL raliGetImageNameLen(RaliContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageNameLen")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
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

void
RALI_API_CALL raliGetImageLabels(RaliContext p_context, int* buf)
{

    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageLabels")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second) {
        WRN("No label has been loaded for this output image")
        return;
    }
    size_t meta_data_batch_size = meta_data.second->get_label_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    memcpy(buf, meta_data.second->get_label_batch().data(),  sizeof(int)*meta_data_batch_size);
}

unsigned
RALI_API_CALL raliGetBoundingBoxCount(RaliContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxCount")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
        THROW("No label has been loaded for this output image")
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        buf[i] = meta_data.second->get_bb_labels_batch()[i].size();
        size += buf[i];
    }
    return size;
}

void
RALI_API_CALL raliGetBoundingBoxLabel(RaliContext p_context, int* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxLabel")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_labels_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        unsigned bb_count = meta_data.second->get_bb_labels_batch()[i].size();
        memcpy(buf, meta_data.second->get_bb_labels_batch()[i].data(),  sizeof(int) * bb_count);
        buf += bb_count;
    }
}



void
RALI_API_CALL raliGetOneHotImageLabels(RaliContext p_context, int* buf, int numOfClasses)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetOneHotImageLabels")
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
RALI_API_CALL raliGetBoundingBoxCords(RaliContext p_context, float* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetBoundingBoxCords")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.second->get_bb_cords_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        unsigned bb_count = meta_data.second->get_bb_cords_batch()[i].size();
        memcpy(buf, meta_data.second->get_bb_cords_batch()[i].data(), bb_count * sizeof(BoundingBoxCord));
        buf += (bb_count * 4);
    }
}

void
RALI_API_CALL raliGetImageSizes(RaliContext p_context, int* buf)
{
    if (!p_context)
        THROW("Invalid rali context passed to raliGetImageSizes")
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
        memcpy(buf, meta_data.second->get_img_sizes_batch()[i].data(), sizeof(ImgSize));
        buf += 2;
    }
}

