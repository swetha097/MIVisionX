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
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ColorCastNode : public Node
{
public:
    ColorCastNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs);
    ColorCastNode() = delete;

    void init( float red, float blue,float green , float alpha, int layout);
    void init( FloatParam* red_param, FloatParam* green_param,  FloatParam* blue_param,  FloatParam* alpha_param, int layout);

protected:
    void create_node() override ;
    void update_node() override;
private:

    ParameterVX<float> _red;
    ParameterVX<float> _green;
    ParameterVX<float> _blue;
    ParameterVX<float> _alpha;

    unsigned _layout, _roi_type;
    constexpr static float RED_RANGE [2] = {0, 100};
    constexpr static float   GREEN_RANGE [2] = {0, 100};
    constexpr static float BLUE_RANGE [2] = {0, 170};
    constexpr static float ALPHA_RANGE [2] = {0, 1};
};
