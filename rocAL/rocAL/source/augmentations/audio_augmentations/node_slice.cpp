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

#include <vx_ext_rpp.h>
#include "node_slice.h"
#include "exception.h"

SliceNode::SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
    Node(inputs, outputs) { }

void SliceNode::create_node() {
    if(_node)
        return;

    const int buffer_size = _batch_size * _dims_stride;
    _fill_values_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, buffer_size);
    vx_status status = vxAddArrayItems(_fill_values_array, buffer_size, _fill_values_vec.data(), sizeof(vx_float32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtRppSlice) node: " + TOSTR(status));
    vx_scalar normalized_anchor = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_anchor);
    vx_scalar normalized_shape = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_shape);
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    vx_scalar axis_mask = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);
    vx_scalar dims_stride = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_dims_stride);
    _node = vxExtRppSlice(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _dst_tensor_roi, _anchor->handle(),
                          _shape->handle(), _fill_values_array, axis_mask, normalized_anchor , normalized_shape, policy, dims_stride);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the slice node (vxRppSlice) failed: " + TOSTR(status))
}

void SliceNode::update_node() {
    // if fill values passed by user less than what is required, replicate the values
    if (_fill_values.size() == 1) {
        std::fill(_fill_values_vec.begin(), _fill_values_vec.end(), _fill_values[0]);
    } else if (_fill_values.size() < _fill_values_vec.size()) {
        for(unsigned i = 0; i < _batch_size; i++) {
            int idx = i * _dims_stride;
            for(unsigned d = 0; d < _dims_stride; d++) {
                _fill_values_vec[idx + d] = _fill_values[i];
            }
        }
    }
    vx_status status = VX_SUCCESS;
    status = vxCopyArrayRange((vx_array)_fill_values_array, 0, _batch_size * _dims_stride, sizeof(vx_float32), _fill_values_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != 0)
        WRN("ERROR: vxCopyArrayRange failed in the slice node (vxExtRppSlice) node: " + TOSTR(status))
}

void SliceNode::init(Tensor *anchor, Tensor *shape, std::vector<float> &fill_values, std::vector<unsigned> &axes, bool normalized_anchor, bool normalized_shape, RocalOutOfBoundsPolicy policy) {
    _normalized_anchor = normalized_anchor;
    _normalized_shape = normalized_shape;
    _policy = policy;
    _anchor = anchor;
    _shape = shape;
    _fill_values = fill_values;
    _dims_stride = _anchor->info().dims()[1];
    for(unsigned d = 0; d < axes.size(); d++)
        _axis_mask |= (1 << axes[d]);
    _fill_values_vec.resize(_batch_size * _dims_stride);
}