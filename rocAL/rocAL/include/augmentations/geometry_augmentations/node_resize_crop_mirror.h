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
#include "parameter_vx.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class ResizeCropMirrorNode : public Node {
public:
    ResizeCropMirrorNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ResizeCropMirrorNode() = delete;
    ~ResizeCropMirrorNode();
    void init(unsigned dest_width, unsigned dest_height, unsigned crop_width, unsigned crop_height, IntParam *mirror, RocalResizeScalingMode scaling_mode,
              const std::vector<unsigned>& max_size, RocalResizeInterpolationType interpolation_type);
    void init(unsigned dest_width, unsigned dest_height, FloatParam * crop_width, FloatParam * crop_height, IntParam *mirror, RocalResizeScalingMode scaling_mode,
              const std::vector<unsigned>& max_size, RocalResizeInterpolationType interpolation_type);
    unsigned int get_dst_width() { return _outputs[0]->info().max_shape()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_shape()[1]; }
    std::shared_ptr<RocalCropParam> get_crop_param() { return _crop_param; }
    vx_array get_mirror() { return _mirror.default_array(); }
    void adjust_out_roi_size();
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _dst_roi_width , _dst_roi_height;
    std::shared_ptr<RocalCropParam> _crop_param;
    int _interpolation_type;
    ParameterVX<int> _mirror;
    constexpr static int MIRROR_RANGE[2] =  {0, 1};
    RocalResizeScalingMode _scaling_mode;
    unsigned _src_width, _src_height, _dst_width, _dst_height, _out_width, _out_height;
    unsigned _max_width = 0, _max_height = 0;
    std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec;
    void * _crop_coordinates;
    vx_tensor _crop_tensor;
};
