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

#pragma once
#include "node.h"
#include "parameter_vx.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class CropParam;

class ResizeCropMirrorNode : public Node
{
public:
    ResizeCropMirrorNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeCropMirrorNode() = delete;
    void init(unsigned int crop_h, unsigned int crop_w, IntParam *mirror);
    void init( FloatParam *crop_h_factor, FloatParam *crop_w_factor, IntParam *mirror);
    unsigned int get_dst_width() { return _outputs[0]->info().width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().height_single(); }
    std::shared_ptr<RaliCropParam> get_crop_param() { return _crop_param; }
    vx_array get_mirror() { return _mirror.default_array(); }
protected:
    void create_node() override;
    void update_node() override;
private:
    std::shared_ptr<RaliCropParam> _crop_param;
    vx_array _dst_roi_width ,_dst_roi_height;
    ParameterVX<int> _mirror; 
    constexpr static int MIRROR_RANGE [2] =  {0, 1};
};

