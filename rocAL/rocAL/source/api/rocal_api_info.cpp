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

#include "commons.h"
#include "context.h"
#include "rocal_api.h"

// TODO - Remove image related calls
// size_t ROCAL_API_CALL rocalGetImageWidth(RocalImage p_tensor)
// {
//     auto tensor = static_cast<Tensor*>(p_tensor);
//     return tensor->info().width();
// }
// size_t ROCAL_API_CALL rocalGetImageHeight(RocalImage p_tensor)
// {
//     auto tensor = static_cast<Tensor*>(p_tensor);
//     return tensor->info().height_batch();
// }

// size_t ROCAL_API_CALL rocalGetImagePlanes(RocalImage p_tensor)
// {
//     auto tensor = static_cast<Tensor*>(p_tensor);
//     return tensor->info().color_plane_count();
// }

// int ROCAL_API_CALL rocalGetOutputColorFormat(RocalContext p_context)
// {
//     auto context = static_cast<Context*>(p_context);
//     auto translate_color_format = [](RocalColorFormat color_format)
//     {
//         switch(color_format){
//             case RocalColorFormat::RGB24:
//                 return 0;
//             case RocalColorFormat::BGR24:
//                 return 1;
//             case RocalColorFormat::U8:
//                 return 2;
//             case RocalColorFormat::RGB_PLANAR:
//                 return 3;
//             default:
//                 THROW("Unsupported Tensor type" + TOSTR(color_format))
//         }
//     };

//     return translate_color_format(context->master_graph->output_color_format());
// }

// size_t ROCAL_API_CALL rocalGetAugmentationBranchCount(RocalContext p_context)
// {
//     auto context = static_cast<Context*>(p_context);
//     return context->master_graph->augmentation_branch_count();
// }

size_t  ROCAL_API_CALL
rocalGetRemainingImages(RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    size_t count = 0;
    try
    {
        count = context->master_graph->remaining_images_count();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

RocalStatus ROCAL_API_CALL rocalGetStatus(RocalContext p_context)
{
    if(!p_context)
        return ROCAL_CONTEXT_INVALID;
    auto context = static_cast<Context*>(p_context);

    if(context->no_error())
        return ROCAL_OK;

    return ROCAL_RUNTIME_ERROR;
}

const char* ROCAL_API_CALL rocalGetErrorMessage(RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->error_msg();
}
TimingInfo
ROCAL_API_CALL rocalGetTimingInfo(RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    auto info = context->timing();
    //INFO("shuffle time "+ TOSTR(info.shuffle_time)); to display time taken for shuffling dataset
    return {info.image_read_time, info.image_decode_time, info.image_process_time, info.copy_to_output};
}

size_t ROCAL_API_CALL rocalIsEmpty(RocalContext p_context)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalIsEmpty")
    auto context = static_cast<Context*>(p_context);
    size_t ret = 0;
    try
    {
        ret = context->master_graph->empty();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return ret;
}
