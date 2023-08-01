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

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class CropResizeNode : public Node {
public:
    CropResizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    CropResizeNode() = delete;
    ~CropResizeNode();
    void init(float area, float aspect_ratio, float x_center_drift, float y_center_drift);
    void init(FloatParam* area, FloatParam *aspect_ratio, FloatParam * x_drift_factor, FloatParam * y_drift_factor);
    RocalROI *get_src_roi() { return _inputs[0]->info().get_roi();}
    RocalROI *get_dst_roi() { return _outputs[0]->info().get_roi();}
    std::shared_ptr<RocalRandomCropParam> get_crop_param() { return _crop_param; }
protected:
    void create_node() override;
    void update_node() override;
private:
    std::shared_ptr<RocalRandomCropParam> _crop_param;
    void * _crop_coordinates;
    vx_tensor _crop_tensor;
    vx_array _dst_roi_width, _dst_roi_height;
};
