/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

class RicapNode : public Node
{
public:
    RicapNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RicapNode() = delete;
    void init(int beta_param);
    unsigned int get_dst_width() { return _outputs[0]->info().width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().height_single(); }
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
    float get_beta_param() { return _beta_param; }
    void set_beta_param(float beta) { _beta_param = beta; }
    vx_array get_permutute_array_1() { return _permute_array_1; }
    vx_array get_permutute_array_2() { return _permute_array_2; }
    vx_array get_permutute_array_3() { return _permute_array_3; }
    vx_array get_permutute_array_4() { return _permute_array_4; }
    vx_array get_crop_region1() { return _crop_region1; }
    vx_array get_crop_region2() { return _crop_region2; }
    vx_array get_crop_region3() { return _crop_region3; }
    vx_array get_crop_region4() { return _crop_region4; }

protected:
    void create_node() override;
    void update_node() override;
private:
    float _beta_param = 0.3;
    std::vector<uint32_t> _initial_permute_array;
    std::vector<uint32_t> _crop_array1, _crop_array2, _crop_array3, _crop_array4;
    vx_array _permute_array_1, _permute_array_2, _permute_array_3, _permute_array_4;
    vx_array _crop_region1, _crop_region2, _crop_region3, _crop_region4;
    void update_permute_array();
    void update_crop_region(float beta);
    

};