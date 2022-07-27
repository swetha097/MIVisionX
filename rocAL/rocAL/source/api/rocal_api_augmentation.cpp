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
#include "node_resize.h"
#include "node_resize_shorter.h"
#include "node_color_twist.h"
#include "node_crop_mirror_normalize.h"
#include "node_crop.h"
#include "node_copy.h"
#include "node_nop.h"
#include "meta_node_crop_mirror_normalize.h"
#include "node_resize.h"
#include "node_crop.h"

#include "meta_node_resize.h"
#include "meta_node_crop.h"
#include "meta_node_ssd_random_crop.h"

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
            tensor_data_type = RocalTensorDataType::FP32;
            return;
        case ROCAL_FP16:
            tensor_data_type = RocalTensorDataType::FP16;
            return;
        case ROCAL_UINT8:
            tensor_data_type = RocalTensorDataType::UINT8;
            return;
        default:
            THROW("Unsupported Tensor output type" + TOSTR(tensor_output_type))
    }
}


RocalTensor ROCAL_API_CALL
rocalBrightness(
        RocalContext p_context,
        RocalTensor p_input,
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
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

// RocalTensor ROCAL_API_CALL
// rocalGamma(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output,
//         RocalFloatParam p_alpha)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     auto alpha = static_cast<FloatParam*>(p_alpha);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<GammaNode>({input}, {output})->init(alpha);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalGammaFixed(
//         RocalContext p_context,
//         RocalTensor p_input,
//         float alpha,
//         bool is_output)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     try
//     {
//         if(!input || !context)
//             THROW("Null values passed as input")

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<GammaNode>({input}, {output})->init(alpha);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalBrightness(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output,
//         RocalFloatParam p_alpha,
//         RocalFloatParam p_beta)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     auto alpha = static_cast<FloatParam*>(p_alpha);
//     auto beta = static_cast<FloatParam*>(p_beta);
//     try
//     {

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<BrightnessNode>({input}, {output})->init(alpha, beta);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor ROCAL_API_CALL
// rocalBrightnessFixed(
//         RocalContext p_context,
//         RocalTensor p_input,
//         float alpha,
//         float beta,
//         bool is_output)
// {
//     if(!p_input || !p_context)
//         THROW("Null values passed as input")
//     Tensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<Tensor*>(p_input);
//     try
//     {
//         if(!input || !context)
//             THROW("Null values passed as input")

//         output = context->master_graph->create_tensor(input->info(), is_output);

//         context->master_graph->add_tensor_node<BrightnessNode>({input}, {output})->init(alpha, beta);
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

// RocalTensor
// ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, RocalTensorLayout rocal_tensor_layout,
//                                     RocalTensorOutputType rocal_tensor_output_type, unsigned crop_depth, unsigned crop_height,
//                                     unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
//                                     std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror)
// {
//     rocALTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocALTensor*>(p_input);
//     auto mirror = static_cast<IntParam *>(p_mirror);
//     float mean_acutal = 0, std_actual = 0; // Mean of vectors
//     for(unsigned i = 0; i < mean.size(); i++)
//     {
//         mean_acutal += mean[i];
//         std_actual  += std_dev[i];
//     }
//     mean_acutal /= mean.size();
//     std_actual /= std_dev.size();
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
//                 std::cerr<<"\n Setting output type to FP32";
//                 op_tensorDataType = RocalTensorDataType::FP32;
//                 break;
//             case ROCAL_FP16:
//                 std::cerr<<"\n Setting output type to FP16";
//                 op_tensorDataType = RocalTensorDataType::FP16;
//                 break;
//             case ROCAL_UINT8:
//                 std::cerr<<"\n Setting output type to UINT8";
//                 op_tensorDataType = RocalTensorDataType::UINT8;
//                 break;
//             default:
//                 THROW("Unsupported Tensor output type" + TOSTR(rocal_tensor_output_type))
//         }
//         // For the crop mirror normalize resize node, user can create an image with a different width and height
//         rocALTensorInfo output_info = input->info();
//         // output_info.format(op_tensorFormat);
//         output_info.set_data_type(op_tensorDataType);
//         output = context->master_graph->create_tensor(output_info, is_output);
//         // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
//         output->reset_tensor_roi();
//         context->master_graph->add_node<CropMirrorNormalizeTensorNode>({input}, {output})->init(crop_height, crop_width, start_x, start_y, mean_acutal,
//                                                                                         std_actual , mirror,layout );
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what());
//     }

//     return output; // Changed to input----------------IMPORTANT
// }


RocalTensor  ROCAL_API_CALL
rocalCopyTensor(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

// Commented for Now
// RocalTensor  ROCAL_API_CALL
// rocalNopTensor(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     rocALTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocALTensor*>(p_input);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);
//         context->master_graph->add_tensor_node<NopNode>({input}, {output});
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

RocalTensor ROCAL_API_CALL
rocalCrop(RocalContext p_context,
          RocalTensor p_input,
          RocalTensorLayout rocal_tensor_layout,
          RocalTensorOutputType rocal_tensor_output_type,
          bool is_output,
          RocalFloatParam p_crop_width,
          RocalFloatParam p_crop_height,
          RocalFloatParam p_crop_depth,
          RocalFloatParam p_crop_pox_x,
          RocalFloatParam p_crop_pos_y,
          RocalFloatParam p_crop_pos_z)
{
    rocALTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocALTensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        // if(!input || !context || crop_width == 0 || crop_height == 0)
        //     THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocALTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<CropNode>({input}, {output})->init(crop_h, crop_w, x_drift, y_drift);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
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
        output_info.set_tensor_layout(op_tensorLayout);
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
        std::vector<unsigned> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = resize_height;
            out_dims[2] = resize_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = resize_height;
            out_dims[3] = resize_width;
        }
        output_info.set_dims(out_dims);
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
