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

//
// Created by mvx on 3/31/20.
//

#include "commons.h"
#include "context.h"
#include "rocal_api.h"

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
ROCAL_API_CALL rocalCreateVideoLabelReader(RocalContext p_context, const char* source_path, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num) {
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateLabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_video_label_reader(source_path, MetaDataReaderType::VIDEO_LABEL_READER, sequence_length, frame_step, frame_stride, file_list_frame_num);

}

RocalMetaData
ROCAL_API_CALL rocalCreateCOCOReader(RocalContext p_context, const char* source_path, bool is_output, bool is_box_encoder, bool is_box_iou_matcher) {
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateCOCOReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_coco_meta_data_reader(source_path, is_output, MetaDataReaderType::COCO_META_DATA_READER,  MetaDataType::BoundingBox, is_box_encoder, is_box_iou_matcher);
}

RocalMetaData
ROCAL_API_CALL rocalCreateTFReader(RocalContext p_context, const char* source_path, bool is_output,const char* user_key_for_label, const char* user_key_for_filename)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateTFReader")
    auto context = static_cast<Context*>(p_context);
    std::string user_key_for_label_str(user_key_for_label);
    std::string user_key_for_filename_str(user_key_for_filename);


    std::map<std::string, std::string> feature_key_map = {
        {"image/class/label", user_key_for_label_str},
        {"image/filename",user_key_for_filename_str}
    };
    return context->master_graph->create_tf_record_meta_data_reader(source_path , MetaDataReaderType::TF_META_DATA_READER , MetaDataType::Label, feature_key_map);}

RocalMetaData
ROCAL_API_CALL rocalCreateTFReaderDetection(RocalContext p_context, const char* source_path, bool is_output,
    const char* user_key_for_label, const char* user_key_for_text,
    const char* user_key_for_xmin, const char* user_key_for_ymin, const char* user_key_for_xmax, const char* user_key_for_ymax,
    const char* user_key_for_filename)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateTFReaderDetection")
    auto context = static_cast<Context*>(p_context);

    std::string user_key_for_label_str(user_key_for_label);
    std::string user_key_for_text_str(user_key_for_text);
    std::string user_key_for_xmin_str(user_key_for_xmin);
    std::string user_key_for_ymin_str(user_key_for_ymin);
    std::string user_key_for_xmax_str(user_key_for_xmax);
    std::string user_key_for_ymax_str(user_key_for_ymax);
    std::string user_key_for_filename_str(user_key_for_filename);

    std::map<std::string, std::string> feature_key_map = {
        {"image/class/label", user_key_for_label_str},
        {"image/class/text", user_key_for_text_str},
        {"image/object/bbox/xmin", user_key_for_xmin_str},
        {"image/object/bbox/ymin", user_key_for_ymin_str},
        {"image/object/bbox/xmax", user_key_for_xmax_str},
        {"image/object/bbox/ymax", user_key_for_ymax_str},
        {"image/filename",user_key_for_filename_str}
    };

    return context->master_graph->create_tf_record_meta_data_reader(source_path , MetaDataReaderType::TF_DETECTION_META_DATA_READER,  MetaDataType::BoundingBox, feature_key_map);
}

RocalMetaData
ROCAL_API_CALL rocalCreateMXNetReader(RocalContext p_context, const char* source_path, bool is_output)
{
    if (!p_context)
        ERR("Invalid rocal context passed to rocalCreateMXNetReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_mxnet_label_reader(source_path, is_output);

}

RocalMetaData
ROCAL_API_CALL rocalCreateTextFileBasedLabelReader(RocalContext p_context, const char* source_path) {

    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateTextFileBasedLabelReader")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->create_label_reader(source_path, MetaDataReaderType::TEXT_FILE_META_DATA_READER);

}

void
ROCAL_API_CALL rocalGetImageName(RocalContext p_context,  char* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageName")
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
ROCAL_API_CALL rocalGetImageNameLen(RocalContext p_context, int* buf)
{
    unsigned size = 0;
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageNameLen")
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
ROCAL_API_CALL rocalGetImageId(RocalContext p_context,  int* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageId")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    size_t meta_data_batch_size = meta_data.first.size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    for(unsigned int i = 0; i < meta_data_batch_size; i++)
    {
        std::string str_id = meta_data.first[i].erase(0, meta_data.first[i].find_first_not_of('0'));
        buf[i] = stoi(str_id);
    }
}

RocalTensorList
ROCAL_API_CALL rocalGetImageLabels(RocalContext p_context)
{

    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageLabels")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->labels_meta_data();
}

unsigned
ROCAL_API_CALL rocalGetBoundingBoxCount(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetBoundingBoxCount")
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second)
        THROW("No label has been loaded for this output image")
    return meta_data.second->get_batch_object_count();
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
ROCAL_API_CALL rocalGetMatchedIndices(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetMatchedIndices")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->matches_meta_data();
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
    if (dest == 0) // HOST DESTINATION
        memcpy(buf, one_hot_encoded, sizeof(int) * meta_data_batch_size * numOfClasses);
    else
    {
#if ENABLE_HIP
        hipError_t err = hipMemcpy(buf, one_hot_encoded, sizeof(int) * meta_data_batch_size * numOfClasses, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            THROW("Invalid Data Pointer: Error copying to device memory")
#elif ENABLE_OPENCL
        if(clEnqueueWriteBuffer(context->master_graph->get_ocl_cmd_q(), (cl_mem)buf, CL_TRUE, 0, sizeof(int) * meta_data_batch_size * numOfClasses, one_hot_encoded, 0, NULL, NULL) != CL_SUCCESS)
            THROW("Invalid Data Pointer: Error copying to device memory")

#endif
    }
}
#endif

RocalTensorList
ROCAL_API_CALL rocalGetBoundingBoxCords(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetBoundingBoxCords")
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->bbox_meta_data();
}

void
ROCAL_API_CALL rocalGetImageSizes(RocalContext p_context, int* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetImageSizes")
    auto context = static_cast<Context*>(p_context);
    auto img_sizes = context->master_graph->get_image_sizes();
    size_t meta_data_batch_size = img_sizes.size();


    if(img_sizes.size() == 0)
    {
        WRN("No sizes has been loaded for this output image")
        return;
    }
    for(unsigned i = 0; i < meta_data_batch_size; i++)
    {
        memcpy(buf, &(img_sizes[i]), sizeof(ImgSize));
        buf += 3;
    }
}

RocalMetaData
ROCAL_API_CALL rocalCreateTextCifar10LabelReader(RocalContext p_context, const char* source_path, const char* file_prefix) {

    if (!p_context)
        THROW("Invalid rocal context passed to rocalCreateTextCifar10LabelReader")
    auto context = static_cast<Context*>(p_context);

    return context->master_graph->create_cifar10_label_reader(source_path, file_prefix);

}

void
ROCAL_API_CALL rocalGetSequenceStartFrameNumber(RocalContext p_context,  unsigned int* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetSequenceStartFrameNumber")
    auto context = static_cast<Context*>(p_context);
    std::vector<size_t> sequence_start_frame;
    context->master_graph->sequence_start_frame_number(sequence_start_frame);
    std::copy(sequence_start_frame.begin(), sequence_start_frame.end(), buf);
}

void
ROCAL_API_CALL rocalGetSequenceFrameTimestamps(RocalContext p_context,  float* buf)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalGetSequenceFrameTimestamps")
    auto context = static_cast<Context*>(p_context);
    std::vector<std::vector<float>> sequence_frame_timestamps;
    context->master_graph->sequence_frame_timestamps(sequence_frame_timestamps);
    auto sequence_length = sequence_frame_timestamps[0].size();
    for (unsigned int i = 0; i < sequence_frame_timestamps.size(); i++)
    {
        std::copy(sequence_frame_timestamps[i].begin(), sequence_frame_timestamps[i].end(), buf);
        buf = buf + sequence_length;
    }
}

void ROCAL_API_CALL rocalBoxEncoder(RocalContext p_context, std::vector<float>& anchors, float criteria,
                                  std::vector<float> &means, std::vector<float> &stds, bool offset, float scale)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalBoxEncoder")
    auto context = static_cast<Context *>(p_context);
    context->master_graph->box_encoder(anchors, criteria, means, stds, offset, scale);
}

void ROCAL_API_CALL rocalBoxIOUMatcher(RocalContext p_context, std::vector<float>& anchors, float criteria,
                                  float high_threshold, float low_threshold ,  bool allow_low_quality_matches)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalBoxIOUMatcher")
    auto context = static_cast<Context *>(p_context);
    context->master_graph->box_iou_matcher(anchors, criteria, high_threshold, low_threshold, allow_low_quality_matches);
}

// RocalMetaData
// ROCAL_API_CALL rocalCopyEncodedBoxesAndLables(RocalContext p_context, float* boxes_buf, int* labels_buf)
// {
//     if (!p_context)
//         THROW("Invalid rocal context passed to rocalCopyEncodedBoxesAndLables")
//     auto context = static_cast<Context *>(p_context);
//     RocalMetaData output_bbox_and_labels;
//     output_bbox_and_labels.emplace_back(context->master_graph->bbox_labels_meta_data());
//     output_bbox_and_labels.emplace_back(context->master_graph->bbox_meta_data());

//     return output_bbox_and_labels;
// }

RocalMetaData
ROCAL_API_CALL rocalGetEncodedBoxesAndLables(RocalContext p_context, int num_encoded_boxes)
{
    if (!p_context) {
        WRN("rocalGetEncodedBoxesAndLables::Invalid context")
    }
    auto context = static_cast<Context *>(p_context);
    return context->master_graph->get_bbox_encoded_buffers(num_encoded_boxes);
}

