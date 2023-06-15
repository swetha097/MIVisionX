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
#include "node_normal_distribution.h"
#include "exception.h"
#include "parameter_factory.h"

NormalDistributionNode::NormalDistributionNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void NormalDistributionNode::create_node()
{
    if(_node)
        return;
    _stride = (vx_size *)malloc(_num_of_dims * sizeof(float));
    _stride[0] = sizeof(float);
    _stride[1] = _stride[0] * _outputs[0]->info().dims()[0];
    _stride[2] = _stride[1] * _outputs[0]->info().dims()[1];
    vx_status status;

    for(uint i = 0; i < _batch_size; i++) {
    update_param();
    _normal_distribution_array[i] = _dist_normal(_rngs[i]);
    }
    _outputs[0]->swap_handle((void *)_normal_distribution_array.data());

}

void NormalDistributionNode::update_node()
{
    vx_status status;

    for(uint i = 0; i < _batch_size; i++) {
    update_param();
    _normal_distribution_array[i] = _dist_normal(_rngs[i]);
    }
 if(status != 0)
        THROW("ERROR: vxCopyArrayRange failed in the pad node (vxExtrppNode_Slice)  node: "+ TOSTR(status))
}

void NormalDistributionNode::update_param()
{
    std::normal_distribution<float> dist_normal(_mean, _std_dev);
    _dist_normal = dist_normal;
}

void NormalDistributionNode::init(float mean, float std_dev) {
    _mean = mean;
    _std_dev = std_dev;
    _num_of_dims = _outputs[0]->info().num_of_dims();
    _normal_distribution_array.resize(_batch_size);
    BatchRNG<std::mt19937> _rng = {ParameterFactory::instance()->get_seed_from_seedsequence(), _batch_size};
    _rngs =_rng;
    update_param();
}