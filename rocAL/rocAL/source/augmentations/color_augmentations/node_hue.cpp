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
#include <graph.h>
#include "node_hue.h"
#include "exception.h"


HueNode::HueNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs),
        _hue(HUE_RANGE[0], HUE_RANGE[1])
{
}

void HueNode::create_node()
{
    if(_node)
        return;

    _hue.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    std::cerr<<"layouttttttttttttttttt"<<_layout<<"\n\n\n\n";
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_Hue(_graph->get(), _inputs[0]->handle(),  _src_tensor_roi, _outputs[0]->handle(), _hue.default_array(), layout, roi_type, _batch_size);
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Hue (vxExtrppNode_HueCorrectionbatchPD) node failed: "+ TOSTR(status))

}

void HueNode::init(float hue, int layout)
{
    std::cerr<<"HueNode::init\n\n";
    _hue.set_param(hue);
    _layout=layout;
    _roi_type = 0;
}

void HueNode::init(FloatParam* hue, int layout)
{
    _hue.set_param(core(hue));
    _layout=layout;
    _roi_type = 0;
}

void HueNode::update_node()
{
     _hue.update_array();
}
