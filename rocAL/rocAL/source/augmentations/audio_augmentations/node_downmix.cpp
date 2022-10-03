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
#include "node_downmix.h"
#include "exception.h"

DownmixNode::DownmixNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void DownmixNode::create_node()
{
    if(_node)
        return;

    auto audio_roi = _inputs[0]->info().get_roi();
    _src_samples.resize(_batch_size);
    _src_channels.resize(_batch_size);
    for (uint i=0; i < _batch_size; i++)
    {
        _src_samples[i] = audio_roi->at(i).x1;
        _src_channels[i] = audio_roi->at(i).y1;
    }
    _src_samples_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _src_channels_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_src_samples_array, _batch_size, _src_samples.data(), sizeof(vx_int32));
    status |= vxAddArrayItems(_src_channels_array, _batch_size, _src_channels.data(), sizeof(vx_int32));

    if(status != 0)
        THROW(" vxAddArrayItems failed in the downmix node (vxExtrppNode_Downmix)  node: "+ TOSTR(status) + "  "+ TOSTR(status))
    vx_scalar normalize_weights = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalize_weights);
    _node = vxExtrppNode_Downmix(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_samples_array, _src_channels_array, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Downmix) node failed: "+ TOSTR(status))

}

void DownmixNode::init(bool normalize_weights)
{
    _normalize_weights = normalize_weights;
}