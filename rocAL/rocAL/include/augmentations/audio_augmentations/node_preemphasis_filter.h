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
#include "graph.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "rocal_api_types.h"

class PreemphasisFilterNode : public Node
{
public:
    PreemphasisFilterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    PreemphasisFilterNode() = delete;
    void init(FloatParam* preemph_coeff, RocalAudioBorderType preemph_border);

protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array _src_samples_size_array;
    std::vector<int> _src_samples_size;
    ParameterVX<float> _preemph_coeff;
    constexpr static float PREEMPH_COEFF_RANGE [2] = {0.97, 0.97}; // 0.97 is the default value set in DALI
    RocalAudioBorderType _preemph_border;
    std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec;
};
