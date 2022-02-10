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


#include "node_gamma.h"
#include "node_brightness.h"
#include "node_crop_mirror_normalize.h"
#include "node_copy.h"
#include "node_nop.h"
#include "meta_node_crop_mirror_normalize.h"

#include "commons.h"
#include "context.h"
#include "rali_api.h"

RaliTensor RALI_API_CALL
raliGamma(
        RaliContext p_context,
        RaliTensor p_input,
        bool is_output,
        RaliFloatParam p_alpha)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_tensor_node<GammaTensorNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliTensor RALI_API_CALL
raliGammaFixed(
        RaliContext p_context,
        RaliTensor p_input,
        float alpha,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_tensor_node<GammaTensorNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliTensor RALI_API_CALL
raliBrightness(
        RaliContext p_context,
        RaliTensor p_input,
        bool is_output,
        RaliFloatParam p_alpha,
        RaliFloatParam p_beta)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try
    {

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_tensor_node<BrightnessTensorNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliTensor RALI_API_CALL
raliBrightnessFixed(
        RaliContext p_context,
        RaliTensor p_input,
        float alpha,
        float beta,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")

        output = context->master_graph->create_tensor(input->info(), is_output);

        context->master_graph->add_tensor_node<BrightnessTensorNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliTensor
RALI_API_CALL raliCropMirrorNormalizeTensor(RaliContext p_context, RaliTensor p_input, RaliTensorLayout rali_tensor_layout,
                                    RaliTensorOutputType rali_tensor_output_type, unsigned crop_depth, unsigned crop_height,
                                    unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
                                    std::vector<float> &std_dev, bool is_output, RaliIntParam p_mirror)
{
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    float mean_acutal = 0, std_actual = 0; // Mean of vectors
    for(unsigned i = 0; i < mean.size(); i++)
    {
        mean_acutal += mean[i];
        std_actual  += std_dev[i];
    }
    mean_acutal /= mean.size();
    std_actual /= std_dev.size();
    RaliTensorFormat op_tensorFormat;
    RaliTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        switch(rali_tensor_layout)
        {
            case 0:
                op_tensorFormat = RaliTensorFormat::NHWC;
                break;
            case 1:
                op_tensorFormat = RaliTensorFormat::NCHW;
                break;
            default:
                THROW("Unsupported Tensor layout" + TOSTR(rali_tensor_layout))
        }

        switch(rali_tensor_output_type)
        {
            case RALI_FP32:
                // std::cerr<<"\n Setting output type to FP32";
                op_tensorDataType = RaliTensorDataType::FP32;
                break;
            case RALI_FP16:
                op_tensorDataType = RaliTensorDataType::FP16;
                break;
            case RALI_UINT8:
                op_tensorDataType = RaliTensorDataType::UINT8;
                break;
            default:
                THROW("Unsupported Tensor output type" + TOSTR(rali_tensor_output_type))
        }
        // For the crop mirror normalize resize node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.width(crop_width);
        output_info.height(crop_height);
        output_info.format(op_tensorFormat);
        output_info.data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();

        context->master_graph->add_tensor_node<CropMirrorNormalizeTensorNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y, mean_acutal,
                                                                                        std_actual , mirror );
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }

    return output; // Changed to input----------------IMPORTANT
}


RaliTensor  RALI_API_CALL
raliCopyTensor(
        RaliContext p_context,
        RaliTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_tensor_node<CopyTensorNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliTensor  RALI_API_CALL
raliNopTensor(
        RaliContext p_context,
        RaliTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_tensor_node<NopTensorNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
