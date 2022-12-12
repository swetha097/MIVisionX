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

#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "parameter_rocal_crop.h"
#include "commons.h"

void RocalCropParam::set_crop_height_factor(Parameter<float>* crop_h_factor)
{
    if(!crop_h_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_height_factor);
    crop_height_factor = crop_h_factor;
}

void RocalCropParam::set_crop_width_factor(Parameter<float>* crop_w_factor)
{
    if(!crop_w_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_width_factor);
    crop_width_factor = crop_w_factor;
}

void RocalCropParam::update_array()
{
    std::cerr << "UPDATE ARRAY!\n";
    fill_crop_dims();
    update_crop_array();
}

void RocalCropParam::fill_crop_dims()
{
    std::cerr << "FILL CROP DIMS\n";
    auto input_roi = in_roi;
    for(uint img_idx =0; img_idx < batch_size; img_idx++)
    {
        if(!(_random))
        {
            // Evaluating user given crop
            cropw_arr_val[img_idx] = (crop_w > input_roi[2]) ? input_roi[2] : crop_w;
            croph_arr_val[img_idx] = (crop_h > input_roi[3]) ? input_roi[3] : crop_h;
            if(_is_center_crop)
            {
                float x_drift, y_drift;
                x_drift = x_drift_factor->get();
                y_drift = y_drift_factor->get();
                x1_arr_val[img_idx] = static_cast<size_t>(x_drift * (input_roi[2] - cropw_arr_val[img_idx]));
                y1_arr_val[img_idx] = static_cast<size_t>(y_drift * (input_roi[3] - croph_arr_val[img_idx]));
            }
            else
            {
                x1_arr_val[img_idx] = (x1 >= input_roi[2]) ? 0 : x1;
                y1_arr_val[img_idx] = (y1 >= input_roi[3]) ? 0 : y1;
            }
            // std::cerr<<"\n In width:: "<<input_roi[2]<<"\t In height:: "<<input_roi[3];
            // std::cerr<<"\n Crop dims:: "<<x1_arr_val[img_idx]<<" "<<y1_arr_val[img_idx]<<" "<<cropw_arr_val[img_idx]<<" "<<croph_arr_val[img_idx]<<"\n";
        }
        else
        {
            float crop_h_factor_, crop_w_factor_, x_drift, y_drift;
            crop_height_factor->renew();
            crop_h_factor_ = crop_height_factor->get();
            crop_width_factor->renew();
            crop_w_factor_ = crop_width_factor->get();
            cropw_arr_val[img_idx] = static_cast<size_t> (crop_w_factor_ * input_roi[2]);
            croph_arr_val[img_idx] = static_cast<size_t> (crop_h_factor_ * input_roi[3]);
            x_drift_factor->renew();
            y_drift_factor->renew();
            y_drift_factor->renew();
            x_drift = x_drift_factor->get();
            y_drift = y_drift_factor->get();
            x1_arr_val[img_idx] = static_cast<size_t>(x_drift * (input_roi[2]  - cropw_arr_val[img_idx]));
            y1_arr_val[img_idx] = static_cast<size_t>(y_drift * (input_roi[3] - croph_arr_val[img_idx]));
        }
        x2_arr_val[img_idx] = x1_arr_val[img_idx] + cropw_arr_val[img_idx];
        y2_arr_val[img_idx] = y1_arr_val[img_idx] + croph_arr_val[img_idx];
        // Evaluating the crop
        (x2_arr_val[img_idx] > input_roi[2]) ? x2_arr_val[img_idx] = input_roi[2] : x2_arr_val[img_idx] = x2_arr_val[img_idx];
        (y2_arr_val[img_idx] > input_roi[3]) ? y2_arr_val[img_idx] = input_roi[3] : y2_arr_val[img_idx] = y2_arr_val[img_idx];
        input_roi += 4;
    }
}

Parameter<float> *RocalCropParam::default_crop_height_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_HEIGHT_FACTOR_RANGE[0],
                                                                         CROP_HEIGHT_FACTOR_RANGE[1])->core;
}

Parameter<float> *RocalCropParam::default_crop_width_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_WIDTH_FACTOR_RANGE[0],
                                                                         CROP_WIDTH_FACTOR_RANGE[1])->core;
}
