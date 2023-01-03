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
#include "node_uniform_distribution.h"
#include "exception.h"

UniformDistributionNode::UniformDistributionNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void UniformDistributionNode::create_node()
{
    if(_node)
        return;
    _stride = (vx_size *)malloc(_num_of_dims * sizeof(float));
    _stride[0] = sizeof(float);
    _stride[1] = _stride[0] * _outputs[0]->info().dims()[0];
    _stride[2] = _stride[1] * _outputs[0]->info().dims()[1];
    vx_status status;

    // create a uniform distribution
    for(uint i = 0; i < _batch_size; i++) {
    update_param();
    _uniform_distribution_array[i] = _dist_uniform(_generator);
    std::cerr << "\n _uniform_distribution_array :"<< _uniform_distribution_array[i];
    }
    _outputs[0]->swap_handle((void *)_uniform_distribution_array.data());

}

void UniformDistributionNode::update_node()
{

    vx_status status;
    // create a uniform distribution
    for(uint i = 0; i < _batch_size; i++) {
    update_param();
    _uniform_distribution_array[i] = _dist_uniform(_generator);
    std::cerr << "\n _uniform_distribution_array :"<< _uniform_distribution_array[i];
    }
 if(status != 0)
        THROW("ERROR: vxCopyArrayRange failed in the pad node (vxExtrppNode_Slice)  node: "+ TOSTR(status))
}

void UniformDistributionNode::update_param()
{
    std::uniform_real_distribution<float> dist_uniform(_min, _max);
    _dist_uniform = dist_uniform;
}

void UniformDistributionNode::init(std::vector<float> &range) {
    _min = range[0];
    _max = range[1];
    std::cerr << "\n _min in uniform : " << _min;
    std::cerr << "\n _max in uniform: " << _max;
    _num_of_dims = _outputs[0]->info().num_of_dims();
    _uniform_distribution_array.resize(_batch_size);
    update_param();
}