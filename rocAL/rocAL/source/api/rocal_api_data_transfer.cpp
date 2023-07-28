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
#include <opencv2/opencv.hpp>
#if ENABLE_OPENCL
#include "CL/cl.h"
#endif

RocalTensorList ROCAL_API_CALL
rocalGetOutputTensors(
                    RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        return context->master_graph->get_output_tensors();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return nullptr;
    }
    return nullptr;
}

RocalStatus ROCAL_API_CALL
rocalToTensor(RocalContext p_context, void *out_ptr, RocalTensorlayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels, RocalOutputMemType output_mem_type)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == RocalTensorlayout::NHWC) ?  RocalTensorlayout::NHWC : RocalTensorlayout::NCHW;
        auto tensor_output_data_type = (tensor_output_type == ROCAL_FP32) ? RocalTensorDataType::FP32 : RocalTensorDataType::FP16;
        context->master_graph->to_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, tensor_output_data_type, output_mem_type);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

inline void saveRGBImage(const unsigned char* imageData, int width, int height, const std::string& filename) {
    // Create a cv::Mat object from the image data
    cv::Mat rgbImage(height, width, CV_8UC3, (void*)imageData);
    // Save the image to the specified file
    cv::imwrite(filename + "output_image_transfer.png", rgbImage);
    std::cerr << "\n Dumped Images";
}

RocalStatus ROCAL_API_CALL
rocalExternalSourceFeedInput(
        RocalContext p_context,
        std::vector<std::string> input_images_names,
        std::vector<int> labels,
        std::vector<unsigned char *>input_buffer,
        std::vector<unsigned> roi_width,
        std::vector<unsigned> roi_height,
        unsigned int max_width,
        unsigned int max_height,
        int channels,
        RocalExtSourceMode mode,
        RocalTensorLayout layout,
        bool eos)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        ExternalFileMode external_file_mode = (ExternalFileMode) mode;
        RocalTensorlayout format = (RocalTensorlayout) layout;
        // std::cerr << "\n Comes to Transfer start ";
        context->master_graph->feed_external_input(input_images_names, labels, input_buffer,
                                                    roi_width, roi_height, max_width, max_height, channels,
                                                    external_file_mode, format, eos);
        // const size_t image_size = max_width * max_height * 3 * sizeof(unsigned char);
        // std::cerr << "\n Comes to Transfer end ";
        // for(uint i = 0; i < roi_width.size(); i++)
        // {
        //     auto image_row_ptr = input_buffer[i];
        //     saveRGBImage(image_row_ptr, max_width, max_height, std::to_string(i));
        // }
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

void
ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalTensor> &output_images)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalSetOutputs")
    auto context = static_cast<Context *>(p_context);
    for (auto& it : output_images) {
        auto img = static_cast<RocalTensor>(it);
        context->master_graph->set_output(img);
    }
}
