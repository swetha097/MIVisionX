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
#include <VX/vx_compatibility.h>
#include "node_vignette.h"
#include "exception.h"

VignetteNode::VignetteNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs),
        _sdev(SDEV_RANGE[0], SDEV_RANGE[1])
{
}

void VignetteNode::create_node()
{
    if(_node)
        return;

    _sdev.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    std::cerr<<"layouttttttttttttttttt"<<_layout<<"\n\n\n\n";
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    // _node = vxExtrppNode_Vignette(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _sdev.default_array(),  layout, roi_type, _batch_size);
    _node = vxExtrppNode_Vignette(_graph->get(), _inputs[0]->handle(),  _src_tensor_roi, _outputs[0]->handle(), _sdev.default_array(), layout, roi_type, _batch_size);


    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the vignette (vxExtrppNode_VignettebatchPD) node failed: "+ TOSTR(status))
}

void VignetteNode::init(float sdev, int layout)
{
    _sdev.set_param(sdev);
    _layout=layout;
    _roi_type = 0;
}

void VignetteNode::init(FloatParam* sdev, int layout)
{
    _sdev.set_param(core(sdev));
    _layout=layout;
    _roi_type = 0;
}

void VignetteNode::update_node()
{
    _sdev.update_array();
}

