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
#include "node_tensor_mul_scalar.h"
#include "exception.h"

TensorMulScalarNode::TensorMulScalarNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void TensorMulScalarNode::create_node()
{
    if(_node)
        return;
    vx_scalar scalar_value = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_scalar);

    _node = vxExtrppNode_TensorMulScalar(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), scalar_value, _src_tensor_roi, _dst_tensor_roi, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the (vxExtrppNode_TensorMulScalar) node failed: "+ TOSTR(status))

}

void TensorMulScalarNode::update_node()
{
}

void TensorMulScalarNode::init(float scalar)
{
    _scalar = scalar;
}