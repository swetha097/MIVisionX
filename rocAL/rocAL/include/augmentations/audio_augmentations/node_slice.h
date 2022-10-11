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
#include "graph.h"
#include "rocal_api_types.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class SliceNode : public Node
{
public:
    SliceNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    SliceNode() = delete;
    void init( FloatParam* anchor_param, FloatParam* shape_param, FloatParam* fill_values_param, int axes,
                bool normalized_anchor, bool normalized_shape, RocalOutOfBoundsPolicy policy);

protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _anchor;
    ParameterVX<float> _shape;
    ParameterVX<float> _fill_values;
    bool _normalized_anchor = false;
    bool _normalized_shape = false;
    RocalOutOfBoundsPolicy _policy = RocalOutOfBoundsPolicy::ERROR;
    int _axes = 0;
    constexpr static int ANCHOR_RANGE [2] = {1, 100}; // Shobi Need to change
    constexpr static int SHAPE_RANGE [2] = {1, 100};
    constexpr static float FILL_VALUES_RANGE [2] = {0, 0};

};