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
#include "node_exposure.h"
#include "node_resize.h"
#include "node_color_cast.h"
#include "node_brightness.h"
#include "node_resize.h"
#include "node_resize_shorter.h"
#include "node_color_twist.h"
#include "node_crop_mirror_normalize.h"
#include "node_crop.h"
#include "node_copy.h"
#include "node_nop.h"
#include "node_resize.h"
#include "node_crop.h"

#include "meta_node_resize.h"
#include "meta_node_crop.h"
#include "meta_node_ssd_random_crop.h"
#include "node_spatter.h"
#include "node_contrast.h"
#include "node_resize_mirror_normalize.h"
#include "node_flip.h"
#include "node_color_jitter.h"
#include "node_noise.h"
#include "node_blend.h"
#include "node_gridmask.h"
#include "node_warp_affine.h"
#include "node_blur.h"
#include "node_vignette.h"
#include "node_jitter.h"
#include "node_snow.h"
#include "node_fog.h"
#include "node_saturation.h"
#include "node_hue.h"
#include "node_fisheye.h"
#include "node_color_temperature.h"
#include "node_pixelate.h"
#include "node_lens_correction.h"
#include "node_rain.h"
#include "node_rotate.h"

#include "commons.h"
#include "context.h"
#include "rocal_api.h"

void get_rocal_tensor_layout(RocalTensorLayout &tensor_layout, RocalTensorlayout &op_tensor_layout, int &layout)
{
    switch(tensor_layout)
    {
        case 0:
            op_tensor_layout = RocalTensorlayout::NHWC;
            layout = 0;
            return;
        case 1:
            op_tensor_layout = RocalTensorlayout::NCHW;
            layout = 1;
            return;
        default:
            THROW("Unsupported Tensor layout" + TOSTR(tensor_layout))
    }
}

void get_rocal_tensor_data_type(RocalTensorOutputType &tensor_output_type, RocalTensorDataType &tensor_data_type)
{
    switch(tensor_output_type)
    {
        case ROCAL_FP32:
            std::cerr<<"\n Setting output type to FP32";
            tensor_data_type = RocalTensorDataType::FP32;
            return;
        case ROCAL_FP16:
            tensor_data_type = RocalTensorDataType::FP16;
            return;
        case ROCAL_UINT8:
            std::cerr<<"\n Setting output type to UINT8";
            tensor_data_type = RocalTensorDataType::UINT8;
            return;
        default:
            THROW("Unsupported Tensor output type" + TOSTR(tensor_output_type))
    }
}

RocalTensor ROCAL_API_CALL
rocalBrightness(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorLayout rocal_tensor_layout,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                RocalFloatParam p_alpha,
                RocalFloatParam p_beta)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        std::cerr<<"\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"<<output_info.get_data_type();
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNoise(RocalContext p_context,
           RocalTensor p_input,
           RocalTensorLayout rocal_tensor_layout,
           RocalTensorOutputType rocal_tensor_output_type,
           bool is_output,
           RocalFloatParam noise_p,
           RocalFloatParam salt_p,
           RocalFloatParam noise_v,
           RocalFloatParam salt_v,
           int seed)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto noise_prob = static_cast<FloatParam*>(noise_p);
    auto salt_prob = static_cast<FloatParam*>(salt_p);
    auto noise_value = static_cast<FloatParam*>(noise_v);
    auto salt_value = static_cast<FloatParam*>(salt_v);

    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout = 0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
         rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<NoiseTensorNode>({input}, {output})->init(noise_prob, salt_prob, noise_value ,salt_value,seed, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalContrast(RocalContext p_context,
              RocalTensor p_input,
              RocalTensorLayout rocal_tensor_layout,
              RocalTensorOutputType rocal_tensor_output_type,
              bool is_output,
              RocalFloatParam c_factor,
              RocalFloatParam c_center)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto contrast_factor = static_cast<FloatParam*>(c_factor);
    auto contrast_center = static_cast<FloatParam*>(c_center);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<ContrastNode>({input}, {output})->init(contrast_factor, contrast_center, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor
ROCAL_API_CALL rocalColorCast(RocalContext p_context,
                              RocalTensor p_input,
                              RocalTensorLayout rocal_tensor_layout,
                              RocalTensorOutputType rocal_tensor_output_type,
                              bool is_output,
                              RocalFloatParam R_value,
                              RocalFloatParam G_value,
                              RocalFloatParam B_value,
                              RocalFloatParam alpha_tensor)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto red = static_cast<FloatParam*>(R_value);
    auto green = static_cast<FloatParam*>(G_value);
    auto blue = static_cast<FloatParam*>(B_value);
    auto alpha = static_cast<FloatParam*>(alpha_tensor);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<ColorCastNode>({input}, {output})->init(red, green, blue ,alpha, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposure(RocalContext p_context,
              RocalTensor p_input,
              RocalTensorLayout rocal_tensor_layout,
              RocalTensorOutputType rocal_tensor_output_type,
              bool is_output,
              RocalFloatParam p_alpha)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<ExposureNode>({input}, {output})->init(alpha,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropFixed(RocalContext p_context, 
          RocalTensor p_input,
          RocalTensorLayout rocal_tensor_layout,
          RocalTensorOutputType rocal_tensor_output_type,
          unsigned crop_depth,
          unsigned crop_height,
          unsigned crop_width,
          float start_x,
          float start_y,
          float start_z,
          bool is_output)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<CropNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

RocalTensor ROCAL_API_CALL
rocalSpatter(RocalContext p_context,
             RocalTensor p_input,
             RocalTensorLayout rocal_tensor_layout,
             RocalTensorOutputType rocal_tensor_output_type,
             bool is_output,
             int R_value,
             int G_value,
             int B_value)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<SpatterNode>({input}, {output})->init(R_value, G_value, B_value, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTwist(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorLayout rocal_tensor_layout,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                RocalFloatParam p_alpha,
                RocalFloatParam p_beta,
                RocalFloatParam p_hue,
                RocalFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue ,sat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorJitter(RocalContext p_context,
                 RocalTensor p_input,
                 RocalTensorLayout rocal_tensor_layout,
                 RocalTensorOutputType rocal_tensor_output_type,
                 bool is_output,
                 RocalFloatParam p_alpha,
                 RocalFloatParam p_beta,
                 RocalFloatParam p_hue,
                 RocalFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<ColorJitterNode>({input}, {output})->init(alpha, beta, hue ,sat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

// RocalTensor ROCAL_API_CALL 
// rocalCropMirrorNormalize(RocalContext p_context, 
//                          RocalTensor p_input, 
//                          RocalTensorLayout rocal_tensor_layout,
//                          RocalTensorOutputType rocal_tensor_output_type,
//                          unsigned crop_depth,
//                          unsigned crop_height,
//                          unsigned crop_width, 
//                          float start_x, 
//                          float start_y, 
//                          float start_z, 
//                          std::vector<float> &mean,
//                          std::vector<float> &std_dev, 
//                          bool is_output, 
//                          RocalIntParam p_mirror)
// {
//     rocALTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocALTensor*>(p_input);
//     auto mirror = static_cast<IntParam *>(p_mirror);
//     RocalTensorlayout op_tensorFormat;
//     RocalTensorDataType op_tensorDataType;
//     try
//     {
//         if(!input || !context || crop_width == 0 || crop_height == 0)
//             THROW("Null values passed as input")
//         int layout=0;
//         switch(rocal_tensor_layout)
//         {
//             case 0:
//                 op_tensorFormat = RocalTensorlayout::NHWC;
//                 layout=0;
//                 std::cerr<<"RocalTensorlayout::NHWC";
//                 break;
//             case 1:
//                 op_tensorFormat = RocalTensorlayout::NCHW;
//                 layout=1;
//                 std::cerr<<"RocalTensorlayout::NCHW";

//                 break;
//             default:
//                 THROW("Unsupported Tensor layout" + TOSTR(rocal_tensor_layout))
//         }
//         switch(rocal_tensor_output_type)
//         {
//             case ROCAL_FP32:
//                 op_tensorDataType = RocalTensorDataType::FP32;
//                 break;
//             case ROCAL_FP16:
//                 op_tensorDataType = RocalTensorDataType::FP16;
//                 break;
//             case ROCAL_UINT8:
//                 op_tensorDataType = RocalTensorDataType::UINT8;
//                 break;
//             default:
//                 THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
//         }
//         // For the crop mirror normalize resize node, user can create an image with a different width and height
//         rocALTensorInfo output_info = input->info();
//         output_info.set_data_type(op_tensorDataType);
//         output = context->master_graph->create_tensor(output_info, is_output);
//         output->reset_tensor_roi();
//         context->master_graph->add_node<CropMirrorNormalizeTensorNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y, mean,
//                                                                                                 std_dev , mirror,layout );
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what());
//     }
//     return output; // Changed to input----------------IMPORTANT
// }
RocalTensor  ROCAL_API_CALL
rocalResizeShorter(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        unsigned size,
        bool is_output)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")
        // For the resize node, user can create an image with a different width and height
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        if (size == 0) size = input->info().max_dims()[0];
        if (size == 0) size = input->info().max_dims()[1];
        std::vector<unsigned> out_dims = output_info.dims();
        int size_dim = size * 10;
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = size_dim;
            out_dims[2] = size_dim;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = size_dim;
            out_dims[3] = size_dim;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeShorterNode> resize_node =  context->master_graph->add_node<ResizeShorterNode>({input}, {output});
        resize_node->init(size);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<ResizeMetaNode,ResizeShorterNode>(resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor  ROCAL_API_CALL
rocalCropCenterFixed(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        // output_info.width(crop_width);
        // output_info.height(crop_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<CropNode>({input}, {output})->init(crop_height, crop_width, layout);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }

    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor
ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, RocalTensorLayout rocal_tensor_layout,
                                    RocalTensorOutputType rocal_tensor_output_type, unsigned crop_depth, unsigned crop_height,
                                    unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
                                    std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")

        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        // output_info.format(op_tensorFormat);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y, mean, std_dev , mirror,layout );
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}
// RocalTensor  ROCAL_API_CALL
// rocalCopyTensor(RocalContext p_context,
//                 RocalTensor p_input,
//                 RocalTensorLayout rocal_tensor_layout,
//                 RocalTensorOutputType rocal_tensor_output_type,
//                 bool is_output)
// {
//     rocALTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocALTensor*>(p_input);
//     RocalTensorlayout op_tensorLayout;
//     RocalTensorDataType op_tensorDataType;
//     try
//     {
//         if(!input || !context)
//             THROW("Null values passed as input")
//         // For the resize node, user can create an image with a different width and height
//         int layout=0;
//         get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
//         get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
//         rocALTensorInfo output_info = input->info();
//         output_info.set_tensor_layout(op_tensorLayout);
//         output_info.set_data_type(op_tensorDataType);
//         // if (size == 0) size = input->info().max_dims()[0];
//         // if (size == 0) size = input->info().max_dims()[1];
//         std::vector<unsigned> out_dims = output_info.dims();
//         int size_dim = size * 10;
//         if(op_tensorLayout == RocalTensorlayout::NHWC)
//         {
//             out_dims[1] = size_dim;
//             out_dims[2] = size_dim;
//         }
//         else if(op_tensorLayout == RocalTensorlayout::NCHW)
//         {
//             out_dims[2] = size_dim;
//             out_dims[3] = size_dim;
//         }
//         output_info.set_dims(out_dims);
//         output = context->master_graph->create_tensor(output_info, is_output);
//         output->reset_tensor_roi();
//         std::shared_ptr<ResizeShorterNode> resize_node =  context->master_graph->add_node<ResizeShorterNode>({input}, {output});
//         resize_node->init(size);
//         // if (context->master_graph->meta_data_graph())
//         //     context->master_graph->meta_add_node<ResizeMetaNode,ResizeShorterNode>(resize_node);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }
RocalTensor ROCAL_API_CALL
rocalGamma(RocalContext p_context,
           RocalTensor p_input,
           RocalTensorLayout rocal_tensor_layout,
           RocalTensorOutputType rocal_tensor_output_type,
           bool is_output,
           RocalFloatParam p_alpha)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<GammaNode>({input}, {output})->init(alpha,layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

//resize
RocalTensor ROCAL_API_CALL 
rocalResize(RocalContext p_context, 
            RocalTensor p_input,
            RocalTensorLayout rocal_tensor_layout,
            RocalTensorOutputType rocal_tensor_output_type,
            unsigned resize_depth,
            unsigned resize_height,
            unsigned resize_width,
            int interpolation_type,
            bool is_output)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || resize_width == 0 || resize_height == 0)
            THROW("Null values passed as input")

        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::cerr<<"resize width and resize height *******************************\n\n"<<resize_height<<"  "<<resize_width;
        output_info.set_width(resize_width);
        output_info.set_height(resize_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<ResizeNode>({input}, {output})->init( interpolation_type, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

//resizemirrornormalize
RocalTensor
ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, 
                                          RocalTensor p_input,
                                          RocalTensorLayout rocal_tensor_layout,
                                          RocalTensorOutputType rocal_tensor_output_type,
                                          unsigned resize_depth,
                                          unsigned resize_height,
                                          unsigned resize_width,
                                          int interpolation_type,
                                          std::vector<float> &mean,
                                          std::vector<float> &std_dev,
                                          bool is_output,
                                          RocalIntParam p_mirror)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || resize_width == 0 || resize_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output_info.set_width(resize_width);
        output_info.set_height(resize_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<ResizeMirrorNormalizeNode>({input}, {output})->init( interpolation_type, mean,std_dev , mirror, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}

RocalTensor ROCAL_API_CALL
rocalFlip(RocalContext p_context,
          RocalTensor p_input,
          RocalTensorLayout rocal_tensor_layout,
          RocalTensorOutputType rocal_tensor_output_type,
          bool is_output,
          RocalIntParam h_flag,
          RocalIntParam v_flag)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto horizontal_flag = static_cast<IntParam*>(h_flag);
    auto vertical_flag = static_cast<IntParam*>(v_flag);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<FlipNode>({input}, {output})->init(horizontal_flag, vertical_flag, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlend(RocalContext p_context,
           RocalTensor p_input,
           RocalTensor p_input1,
           RocalTensorLayout rocal_tensor_layout,
           RocalTensorOutputType rocal_tensor_output_type,
           bool is_output,
           RocalFloatParam p_alpha)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto input1 = static_cast<rocALTensor*>(p_input1);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<BlendNode>({input,input1}, {output})->init(alpha, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGridmask(RocalContext p_context,
              RocalTensor p_input,
              RocalTensorLayout rocal_tensor_layout,
              RocalTensorOutputType rocal_tensor_output_type,
              bool is_output,
              int tileWidth,
              float gridRatio,
              float gridAngle,
              unsigned int shift_x,
              unsigned int shift_y )
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout = 0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
         rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<GridmaskNode>({input}, {output})->init(tileWidth, gridRatio, gridAngle,shift_x,shift_y, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor
ROCAL_API_CALL rocalWarpAffine(RocalContext p_context,
                              RocalTensor p_input,
                              RocalTensorLayout rocal_tensor_layout,
                              RocalTensorOutputType rocal_tensor_output_type,
                              bool is_output,
                              RocalFloatParam x0,
                              RocalFloatParam x1,
                              RocalFloatParam y0,
                              RocalFloatParam y1,
                              RocalFloatParam o0,
                              RocalFloatParam o1,
                              int interpolation_type)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto _x0 = static_cast<FloatParam*>(x0);
    auto _x1 = static_cast<FloatParam*>(x1);
    auto _y0 = static_cast<FloatParam*>(y0);
    auto _y1 = static_cast<FloatParam*>(y1);
    auto _o0 = static_cast<FloatParam*>(o0);
    auto _o1 = static_cast<FloatParam*>(o1);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        std::cerr<<"In rocal_api_augmentation/n/n";
        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(_x0,_x1,_y0,_y1,_o0,_o1, interpolation_type, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



//batchpd tensor support 
RocalTensor ROCAL_API_CALL
rocalBlur(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalIntParam p_sdev)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto sdev = static_cast<IntParam*>(p_sdev);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " BLUR LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<BlurNode>({input}, {output})->init(sdev, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



RocalTensor ROCAL_API_CALL
rocalVignette(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalIntParam p_sdev)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto sdev = static_cast<FloatParam*>(p_sdev);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " VIGNETTE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalJitter(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalIntParam p_sdev)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto sdev = static_cast<IntParam*>(p_sdev);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " JITTER LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<JitterNode>({input}, {output})->init(sdev, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalSnow(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_snow)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto snow_prob = static_cast<FloatParam*>(p_snow);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " SNOW LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<SnowNode>({input}, {output})->init(snow_prob, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFog(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_fog)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto fog_prob = static_cast<FloatParam*>(p_fog);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " SNOW LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<FogNode>({input}, {output})->init(fog_prob, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



RocalTensor ROCAL_API_CALL
rocalSaturation(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_sat)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto sat = static_cast<FloatParam*>(p_sat);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " SNOW LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<SatNode>({input}, {output})->init(sat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalHue(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_hue)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto hue = static_cast<FloatParam*>(p_hue);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " HUE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::cerr<<"###########################innnnn\n\n";
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue, layout);
        std::cerr<<"helloooo\n\n";
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFisheye(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " FISHEYE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<FisheyeNode>({input}, {output})->init( layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTemperature(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalIntParam p_adjust_value)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto adjust_value = static_cast<IntParam*>(p_adjust_value);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " COLORTEMPERATURE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adjust_value, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalRain(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_percentage,
        RocalIntParam p_width,
        RocalIntParam p_height,
        RocalFloatParam p_tranparency)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto rain_percentage = static_cast<FloatParam*>(p_percentage);
    auto width = static_cast<IntParam*>(p_width);
    auto height  = static_cast<IntParam*>(p_height);
    auto rain_transpancy = static_cast<FloatParam*>(p_tranparency);


    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " RAIN LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_percentage,width,height,rain_transpancy, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



RocalTensor ROCAL_API_CALL
rocalLensCorrection(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        RocalFloatParam p_strength,
        RocalFloatParam p_zoom)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto strength = static_cast<FloatParam*>(p_strength);
    auto zoom = static_cast<FloatParam*>(p_zoom);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " LENSCORRECTION LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(strength,zoom, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalPixelate(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " PIXELATE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<PixelateNode>({input}, {output})->init( layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalRotate(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        unsigned dest_width,
        unsigned dest_height,
        int outputformat,
        RocalFloatParam p_angle)
{
    rocALTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto angle = static_cast<FloatParam*>(p_angle);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout );
        std::cerr << " ROTATE LAYOUT : " << layout << "\n";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        std::cerr<<"op_tensorDataType"<<(unsigned)op_tensorDataType;
        output_info.set_data_type(op_tensorDataType);
        output_info.set_width(dest_width);
        output_info.set_height(dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        
        if(dest_width != 0 && dest_height != 0)
             output->reset_tensor_roi();
        

        context->master_graph->add_node<RotateNode>({input}, {output})->init(angle,outputformat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
