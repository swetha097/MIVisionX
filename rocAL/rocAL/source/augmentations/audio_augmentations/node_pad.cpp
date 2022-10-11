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
    std::vector<int> src_frames(_batch_size, _inputs[0]->info().max_dims()[0]);
    std::vector<int> src_channels(_batch_size, _inputs[0]->info().max_dims()[1]);

    _src_frames_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _src_channels_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_src_frames_array, _batch_size, src_frames.data(), sizeof(vx_int32));
    status |= vxAddArrayItems(_src_channels_array, _batch_size, src_channels.data(), sizeof(vx_int32));

    if(status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtrppNode_Normalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))

    // Slice Node To be called
    // _node = vxExtrppNode_Normalize(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_frames_array, _src_channels_array, _axis_mask, _mean, _std_dev,
    //                                _scale, _shift, _epsilon, _ddof, _num_of_dims, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Downmix) node failed: "+ TOSTR(status))

}

void PadNode::update_node() {
    auto audio_roi = _inputs[0]->info().get_roi();
    bool has_same_dim = true;
    for (uint i=0; i < _batch_size; i++) {
        _src_frames[i] = audio_roi->at(i).x1;
        _src_channels[i] = audio_roi->at(i).y1;
        // TODO - Need to update the anchors
        // TODO - Need to update dst the shpae with the max dims of output tensor
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

void PadNode::init(float fill_value) {
    _fill_value = fill_value;
}