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
#include "node_normalize.h"
#include "exception.h"

NormalizeNode::NormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs) {
}

void NormalizeNode::create_node() {
    if(_node)
        return;
    std::vector<int> src_frames(_batch_size, _inputs[0]->info().max_dims()[0]);
    std::vector<int> src_channels(_batch_size, _inputs[0]->info().max_dims()[1]);

    _src_frames_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _src_channels_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_src_frames_array, _batch_size, src_frames.data(), sizeof(vx_int32));
    status |= vxAddArrayItems(_src_channels_array, _batch_size, src_channels.data(), sizeof(vx_int32));

    if(status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtrppNode_Normalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))
    vx_scalar mean = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_mean);
    vx_scalar std_dev = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_std_dev);
    vx_scalar scale = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_scale);
    vx_scalar shift = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_shift);
    vx_scalar epsilon = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_epsilon);
    vx_scalar ddof = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_ddof);
    vx_scalar axis_mask = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);

    _node = vxExtrppNode_Normalize(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_frames_array, _src_channels_array, axis_mask, mean, std_dev,
                                   scale, shift, epsilon, ddof, _num_of_dims, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Downmix) node failed: "+ TOSTR(status))

}

void NormalizeNode::update_node() {
    auto audio_roi = _inputs[0]->info().get_roi();
    bool has_same_dim = true;
    for (uint i=0; i < _batch_size; i++) {
        _src_frames[i] = audio_roi[i].y1;
        _src_channels[i] = audio_roi[i].x1;
        _dst_roi_width_vec[i] =audio_roi[i].y1;
        _dst_roi_height_vec[i] =  audio_roi[i].x1;

    }

    if(!has_same_dim && _batch_size)
        THROW("All the tensor must have same dimension to perform Batch Normalization")

    vx_status status = VX_SUCCESS;
    status |= vxCopyArrayRange((vx_array)_src_frames_array, 0, _batch_size, sizeof(vx_uint32), _src_frames.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyArrayRange((vx_array)_src_channels_array, 0, _batch_size, sizeof(vx_uint32), _src_channels.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    if(status != 0)
        WRN("ERROR: vxCopyArrayRange failed in the normalize node (vxExtrppNode_Normalize)  node: "+ TOSTR(status))
    _src_frames.clear();
    _src_channels.clear();
}

void NormalizeNode::init(float mean, float std_dev, std::vector<int> axes, bool batch,
                        float scale, float shift, int ddof, float epsilon) {
    _mean = mean;
    _std_dev = std_dev;
    _axes = axes;
    _batch = batch;
    _scale = scale;
    _shift = shift;
    _ddof = ddof;
    _epsilon = epsilon;
    if(_inputs[0]->info().num_of_dims() == _outputs[0]->info().num_of_dims())
        _num_of_dims = _inputs[0]->info().num_of_dims() - 1;
    else
        THROW("The input and ouput must have same dimensions")
    _param_shape.resize(_batch_size);
    if(_mean > 0.0f || _std_dev > 0.0f && !_axes.size()) {
        _axes.resize(_num_of_dims);
        std::iota(_axes.begin(), _axes.end(), 0);
    }

    for(int d = 0; d < _axes.size(); d++)
        _axis_mask |= (1 << _axes[d]);

    _src_frames.resize(_batch_size);
    _src_channels.resize(_batch_size);
    _dst_roi_width_vec.resize(_batch_size);
    _dst_roi_height_vec.resize(_batch_size);
}