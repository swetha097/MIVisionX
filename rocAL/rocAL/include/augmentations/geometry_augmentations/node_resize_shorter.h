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

class ResizeShorterNode : public Node
{
public:
    ResizeShorterNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    ResizeShorterNode() = delete;
    void init(int size);
    unsigned int get_dst_width() { return _outputs[0]->info().max_dims()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_dims()[1]; }
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _dst_roi_width , _dst_roi_height ;
    std::vector<uint> _dest_width_val, _dest_height_val;
    int _size;
    unsigned _layout, _roi_type;
    int _interpolation_type;
};
