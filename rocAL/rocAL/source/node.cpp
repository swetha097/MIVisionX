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

#include "node.h"

TensorNode::~TensorNode()
{

    if(!_node)
        vxReleaseNode(&_node);
    _node = nullptr;
}

void
TensorNode::create(std::shared_ptr<Graph> graph)
{
    // if(_outputs.empty() || _inputs.empty())
    //     THROW("Uninitialized input/output images to the node")

    _graph = graph;

    // if(!_inputs.empty())
    // {
        vx_status roi_status;
        std::vector<uint32_t> _src_roi;
        _src_roi.reserve(_batch_size * 4);
        _src_tensor_roi = vxCreateArray(vxGetContext((vx_reference) _graph->get()), VX_TYPE_UINT32, _batch_size * 4);
        roi_status = vxAddArrayItems(_src_tensor_roi, _batch_size * 4, _src_roi.data(), sizeof(vx_uint32));
        if (roi_status != 0)
            THROW(" vxAddArrayItems failed : " + TOSTR(roi_status))
    // }

    create_node();
}

void
TensorNode::update_parameters()
{
    update_node();
    update_src_roi();
}

void
TensorNode::update_src_roi()
{
    if(_inputs[0]->info().is_image())
    {
        vx_status roi_status;
        roi_status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size * 4, sizeof(vx_uint32), _inputs[0]->info().get_roi()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if(roi_status != 0)
            THROW(" Failed calling vxCopyArrayRange for width / height status : "+ TOSTR(roi_status))
    }
}
