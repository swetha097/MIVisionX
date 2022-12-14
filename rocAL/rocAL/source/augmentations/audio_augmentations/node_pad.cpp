
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
#include "node_pad.h"
#include "exception.h"

PadNode::PadNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs) {
}

void PadNode::create_node() {
    if(_node)
        return;
    std::vector<float> anchors(_batch_size * _num_of_dims, 0);
    std::vector<float> shape(_batch_size * _num_of_dims, 0);
    std::vector<float> fill_value(_batch_size * _num_of_dims, 0);
    _stride = (vx_size *)malloc((_num_of_dims ) * tensor_data_size(_outputs[0]->info().data_type()));
    _stride[0] = sizeof(float);
    _stride[1] = _stride[0] * _inputs[0]->info().dims()[0];
    vx_status status;

    _fill_values_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * _num_of_dims);
    std::cerr << "\n _inputs[0]->info().dims().data()" <<_inputs[0]->info().dims()[0];
    _anchors_tensor = vxCreateTensor(vxGetContext((vx_reference)_graph->get()), (_num_of_dims), _outputs[0]->info().dims().data(), VX_TYPE_FLOAT32, 0);
    if ((status = vxGetStatus((vx_reference)_anchors_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor for _anchors_tensor: failed " + TOSTR(status))
    _shapes_tensor = vxCreateTensor(vxGetContext((vx_reference)_graph->get()), (_num_of_dims), _outputs[0]->info().dims().data(), VX_TYPE_FLOAT32, 0);
    if ((status = vxGetStatus((vx_reference)_shapes_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor for _shapes_tensor: failed " + TOSTR(status))

    status = vxAddArrayItems(_fill_values_array, _batch_size * _num_of_dims, anchors.data(), sizeof(vx_float32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtrppNode_Slice) node: "+ TOSTR(status));

    vx_scalar normalized_anchor = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_anchor);
    vx_scalar normalized_shape = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_shape);
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    vx_scalar axis_mask = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);
    _node = vxExtrppNode_Slice(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _anchors_tensor,
                                _shapes_tensor, _fill_values_array, axis_mask, normalized_anchor , normalized_shape, policy, _batch_size);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the pad (vxExtrppNode_Slice) node failed: "+ TOSTR(status))

}

void PadNode::update_node() {
    vx_status src_roi_status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size * 4, sizeof(vx_uint32), _inputs[0]->info().get_roi()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(src_roi_status != 0)
        THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src_roi_status))
    auto audio_roi = _inputs[0]->info().get_roi();
    bool has_same_dim = true;
    std::cerr << "\n Num of dims in pad :" << _num_of_dims;
    for(unsigned i = 0; i < _batch_size; i++) {
        int idx = i * _num_of_dims;
        for(unsigned d = 0; d < _num_of_dims; d++) {
            _anchor_vec[idx + d] = 0;
            _shape_vec[idx + d] = (d == 0) ? audio_roi->at(i).x1 : audio_roi->at(i).y1;
            std::cerr << "\n  _shape_vec[idx + d]  : " << _shape_vec[idx + d];
            _fill_values_vec[idx + d] = _fill_value;
        }
    }

    if(!has_same_dim && _batch_size)
        THROW("All the tensor must have same dimension to perform Batch Normalization")

    vx_status status = VX_SUCCESS;
    status |= vxCopyTensorPatch((vx_tensor)_anchors_tensor, (_num_of_dims), nullptr, nullptr, _stride, _anchor_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyTensorPatch((vx_tensor)_shapes_tensor, (_num_of_dims), nullptr, nullptr, _stride, _shape_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyArrayRange((vx_array)_fill_values_array, 0, _batch_size * _num_of_dims, sizeof(vx_float32), _fill_values_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != 0)
        THROW("ERROR: vxCopyArrayRange failed in the pad node (vxExtrppNode_Slice)  node: "+ TOSTR(status))
    _anchor_vec.clear();
    _shape_vec.clear();
    _fill_values_vec.clear();
}

void PadNode::init(float fill_value) {
    _fill_value = fill_value;
    _num_of_dims = _inputs[0]->info().num_of_dims() - 1;
    _anchor_vec.resize(_batch_size * _num_of_dims);
    _shape_vec.resize(_batch_size * _num_of_dims);
    _fill_values_vec.resize(_batch_size * _num_of_dims);
}