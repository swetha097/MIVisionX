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

#include <vx_ext_rpp.h>
#include "node_slice.h"
#include "exception.h"

SliceNode::SliceNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _anchor(ANCHOR_RANGE[0], ANCHOR_RANGE[1]),
        _shape(SHAPE_RANGE[0], SHAPE_RANGE[1]),
        _fill_values(FILL_VALUES_RANGE[0], FILL_VALUES_RANGE[1])
{
}

void SliceNode::create_node()
{
    if(_node)
        return;

    _anchor.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _shape.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _fill_values.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    vx_scalar normalized_anchor = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_anchor);
    vx_scalar normalized_shape = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_shape);
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    vx_scalar axes = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axes);
    _node = vxExtrppNode_Slice(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _anchor.default_array(),
                                _shape.default_array(), _fill_values.default_array(), axes, normalized_anchor , normalized_shape, policy, _batch_size);
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Slice) node failed: "+ TOSTR(status))

}

void SliceNode::update_node()
{
    std::cerr<<"\n SliceNode::update_node()";
    vx_status src_roi_status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size * 4, sizeof(vx_uint32), _inputs[0]->info().get_roi()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(src_roi_status != 0)
        THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src_roi_status))
}

void SliceNode::init(FloatParam* anchor, FloatParam* shape, FloatParam* fill_values, int axes, bool normalized_anchor, bool normalized_shape, RocalOutOfBoundsPolicy policy)
{
    _anchor.set_param(core(anchor));
    _shape.set_param(core(shape));
    _fill_values.set_param(core(fill_values));
    _axes = axes;
    _normalized_anchor = normalized_anchor;
    _normalized_shape = normalized_shape;
    _policy = policy;
}