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
#include "node_saturation.h"
#include "exception.h"


SaturationNode::SaturationNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs),
        _sat(SAT_RANGE[0], SAT_RANGE[1]) { }

void SaturationNode::create_node() {
    if(_node)
        return;

    _sat.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxRppSaturation(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _sat.default_array(), _input_layout, _output_layout, _roi_type);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the saturation (vxRppGamma) node failed: "+ TOSTR(status))
}

void SaturationNode::init(float sat) {
    _sat.set_param(sat);
}

void SaturationNode::init(FloatParam *sat) {
    _sat.set_param(core(sat));
}

void SaturationNode::update_node() {
    _sat.update_array();
}
