/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include <graph.h>
#include "node_ricap.h"
#include "exception.h"
#include <time.h>
#include <cmath>
#include <random>
#include <boost/math/distributions.hpp> 
#include <boost/math/special_functions/beta.hpp>
using namespace boost::math; 

RicapNode::RicapNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

int inline my_random(int min, int max)
{
  return rand() % (max - min + 1) + min;
}

void RicapNode::create_node()
{
    if(_node)
        return;

    vx_status status_1,status_2,status_3,status_4,status_crop_1,status_crop_2,status_crop_3,status_crop_4;
    _initial_permute_array.resize(_batch_size);
    _crop_array1.resize(4);
    _crop_array2.resize(4);
    _crop_array3.resize(4);
    _crop_array4.resize(4);

    std::random_device rd;
    std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> unif(0,1);
    double p = unif(gen);
    double randFromDist = boost::math::ibeta_inv(_beta_param, _beta_param, p); 
    // Assuming all the input & output images are of the same size 
    uint32_t Ix = get_dst_width();
    uint32_t Iy = get_dst_height();
    // Generating 4 Crop Regions 
    _crop_array1[2] = std::round(randFromDist * Ix); //w1
    _crop_array1[3] = std::round(randFromDist * Iy); //h1
    _crop_array2[2] = Ix - _crop_array1[2]; //w2
    _crop_array2[3] = _crop_array1[3]; //h2
    _crop_array3[2] = _crop_array1[2]; //w3
    _crop_array3[3] = Iy -_crop_array1[3]; //h3
    _crop_array4[2] = Ix -_crop_array1[2]; //w4
    _crop_array4[3] = Iy -_crop_array1[3]; //h4
    _crop_array1[0] = my_random(0, Ix - _crop_array1[2] + 1);// x1
    _crop_array2[0] = my_random(0, Ix - _crop_array2[2] + 1);// x2
    _crop_array3[0] = my_random(0, Ix - _crop_array3[2] + 1);// x3
    _crop_array4[0] = my_random(0, Ix - _crop_array4[2] + 1);// x4
    _crop_array1[1] = my_random(0, Iy - _crop_array1[3] + 1);//y1
    _crop_array2[1] = my_random(0, Iy - _crop_array2[3] + 1);//y2
    _crop_array3[1] = my_random(0, Iy - _crop_array3[3] + 1);//y3
    _crop_array4[1] = my_random(0, Iy - _crop_array4[3] + 1);//y4    

    for (uint i = 0; i < _batch_size; i++ )
    {
        _initial_permute_array[i] = i;
    }

    _permute_array_1 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _permute_array_2 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _permute_array_3 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _permute_array_4 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    _crop_region1 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, 4);
    _crop_region2 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, 4);
    _crop_region3 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, 4);
    _crop_region4 = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, 4);
    
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status_1 = vxAddArrayItems(_permute_array_1,_batch_size , _initial_permute_array.data(), sizeof(vx_uint32));
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status_2 = vxAddArrayItems(_permute_array_2,_batch_size , _initial_permute_array.data(), sizeof(vx_uint32));
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status_3 = vxAddArrayItems(_permute_array_3,_batch_size , _initial_permute_array.data(), sizeof(vx_uint32));
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status_4 = vxAddArrayItems(_permute_array_4,_batch_size , _initial_permute_array.data(), sizeof(vx_uint32));

    status_crop_1 = vxAddArrayItems(_crop_region1, 4 , _crop_array1.data(), sizeof(vx_uint32));
    status_crop_2 = vxAddArrayItems(_crop_region2, 4 , _crop_array2.data(), sizeof(vx_uint32));
    status_crop_3 = vxAddArrayItems(_crop_region3, 4 , _crop_array3.data(), sizeof(vx_uint32));
    status_crop_4 = vxAddArrayItems(_crop_region4, 4 , _crop_array4.data(), sizeof(vx_uint32));
    if(status_crop_1 != 0 || status_crop_2!=0 || status_crop_3!=0 || status_crop_4!=0)
        THROW("\n vxAddArrayItems failed in the Ricap(vxExtrppNode_RicapPD) node: "+ TOSTR(status_crop_1))

    _node = vxExtrppNode_Ricap(_graph->get(), _inputs[0]->handle(), _src_roi_width,_src_roi_height, _outputs[0]->handle(), _permute_array_1, _permute_array_2, _permute_array_3, _permute_array_4, _crop_region1, _crop_region2, _crop_region3, _crop_region4, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ricap (vxExtrppNode_Ricap) node failed: "+ TOSTR(status))

}

void RicapNode::update_permute_array()
{
    vx_status status1, status2, status3, status4;
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status1 = vxCopyArrayRange((vx_array)_permute_array_1, 0, _batch_size , sizeof(vx_uint32), _initial_permute_array.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); 
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status2 = vxCopyArrayRange((vx_array)_permute_array_2, 0, _batch_size , sizeof(vx_uint32), _initial_permute_array.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); 
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status3 = vxCopyArrayRange((vx_array)_permute_array_3, 0, _batch_size , sizeof(vx_uint32), _initial_permute_array.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); 
    std::random_shuffle(_initial_permute_array.begin(), _initial_permute_array.end());
    status4 = vxCopyArrayRange((vx_array)_permute_array_4, 0, _batch_size , sizeof(vx_uint32), _initial_permute_array.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); 

    if(status1 != 0 || status2!=0 || status3!=0 || status4!=0)
        THROW(" vxCopyArrayRange failed in the Ricap(vxExtrppNode_RicapPD) node: "+ TOSTR(status1))
    
}

void RicapNode::update_crop_region(float _beta_param)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> unif(0,1);
    double p = unif(gen);
    double randFromDist = boost::math::ibeta_inv(_beta_param, _beta_param, p); 
    vx_status status1, status2, status3, status4;
    // Assuming all the input & output images are of the same size 
    uint32_t Ix = get_dst_width();
    uint32_t Iy = get_dst_height();
    _crop_array1[2] = std::round(randFromDist * Ix); //w1
    _crop_array1[3] = std::round(randFromDist * Iy); //h1
    _crop_array2[2] = Ix - _crop_array1[2]; //w2
    _crop_array2[3] = _crop_array1[3]; //h2
    _crop_array3[2] = _crop_array1[2]; //w3
    _crop_array3[3] = Iy -_crop_array1[3]; //h3
    _crop_array4[2] = Ix -_crop_array1[2]; //w4
    _crop_array4[3] = Iy -_crop_array1[3]; //h4
    _crop_array1[0] = my_random(0, Ix - _crop_array1[2] + 1);// x1
    _crop_array2[0] = my_random(0, Ix - _crop_array2[2] + 1);// x2
    _crop_array3[0] = my_random(0, Ix - _crop_array3[2] + 1);// x3
    _crop_array4[0] = my_random(0, Ix - _crop_array4[2] + 1);// x4
    _crop_array1[1] = my_random(0, Iy - _crop_array1[3] + 1);//y1
    _crop_array2[1] = my_random(0, Iy - _crop_array2[3] + 1);//y2
    _crop_array3[1] = my_random(0, Iy - _crop_array3[3] + 1);//y3
    _crop_array4[1] = my_random(0, Iy - _crop_array4[3] + 1);//y4
    status1 = vxCopyArrayRange((vx_array)_crop_region1, 0, 4 , sizeof(vx_uint32), _crop_array1.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status2 = vxCopyArrayRange((vx_array)_crop_region2, 0, 4 , sizeof(vx_uint32), _crop_array2.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status3 = vxCopyArrayRange((vx_array)_crop_region3, 0, 4 , sizeof(vx_uint32), _crop_array3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status4 = vxCopyArrayRange((vx_array)_crop_region4, 0, 4 , sizeof(vx_uint32), _crop_array4.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status1 != 0 || status2!=0 || status3!=0 || status4!=0)
        THROW("\n vxCopyArrayRange failed in the Ricap(vxExtrppNode_RicapPD) node: "+ TOSTR(status1))
}

void RicapNode::init(int beta_param)
{
    _beta_param = beta_param;
}

void RicapNode::update_node()
{
    update_permute_array();
    update_crop_region(_beta_param);
}
