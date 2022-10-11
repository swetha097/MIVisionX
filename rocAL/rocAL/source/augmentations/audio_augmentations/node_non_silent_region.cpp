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
#include "node_non_silent_region.h"
#include "exception.h"

NonSilentRegionNode::NonSilentRegionNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void NonSilentRegionNode::create_node()
{
    if(_node)
        return;
    _src_samples_size.resize(_batch_size);
    _src_samples_size_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_src_samples_size_array, _batch_size, _src_samples_size.data(), sizeof(vx_int32));

    vx_scalar cutOffDB = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_cutOffDB);
    vx_scalar reference_power = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_reference_power);
    vx_scalar window_length = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_length);
    vx_scalar reset_interval = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_reset_interval);

    _node = vxExtrppNode_NonSilentRegion(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_samples_size_array,
                                         cutOffDB, reference_power, window_length, reset_interval, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Spectrogram) node failed: "+ TOSTR(status))

}

void NonSilentRegionNode::update_node()
{
    auto audio_roi = _inputs[0]->info().get_roi();
    for (uint i=0; i < _batch_size; i++)
    {
        _src_samples_size[i] = audio_roi->at(i).x1 * audio_roi->at(i).y1;
    }
    vx_status src_roi_status;
    src_roi_status = vxCopyArrayRange((vx_array)_src_samples_size_array, 0, _batch_size, sizeof(vx_uint32), _src_samples_size.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(src_roi_status != 0)
        THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src_roi_status))
}

void NonSilentRegionNode::init(float cutOffDB, float reference_power, int window_length, int reset_interval)
{
    _cutOffDB = cutOffDB;
    _reference_power = reference_power;
    _window_length = window_length;
    _reset_interval = reset_interval;
}