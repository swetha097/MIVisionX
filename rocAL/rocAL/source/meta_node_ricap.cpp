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

#include "meta_node_ricap.h"
void RicapMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _permute_array_val_1.resize(_batch_size);
    _permute_array_val_2.resize(_batch_size);
    _permute_array_val_3.resize(_batch_size);
    _permute_array_val_4.resize(_batch_size);

}
void RicapMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    std::cerr<<"In update params meta_node_ricap.cpp";
    initialize();
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();

    _permute_array_1 = _node->get_permutute_array_1();
    _permute_array_2 = _node->get_permutute_array_2();
    _permute_array_3 = _node->get_permutute_array_3();
    _permute_array_4 = _node->get_permutute_array_4();
    //TODO: Get Crop Coordinates (x,y,w,h) - required to calculate the ratio wrt original Image w & h

    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    // vxCopyArrayRange((vx_array)_flip_axis, 0, _batch_size, sizeof(int),_flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_permute_array_1, 0, _batch_size, sizeof(int),_permute_array_val_1.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_permute_array_2, 0, _batch_size, sizeof(int),_permute_array_val_2.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_permute_array_3, 0, _batch_size, sizeof(int),_permute_array_val_3.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_permute_array_4, 0, _batch_size, sizeof(int),_permute_array_val_4.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);


    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        BoundingBoxLabels labels_buf;
        labels_buf.resize(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        // TODO: Do something with the target labels according to the permuted indices
        // TODO : Return the target labels & crop ratio's
        //TODO: Each image now has 4 target labels according to the patched images (correspoinding indices of the 4 permuted arrays form an image & thus the target labels)
        input_meta_data->get_bb_labels_batch()[i] = labels_buf;
    }
}
