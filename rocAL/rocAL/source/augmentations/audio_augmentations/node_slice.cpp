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
#include "node_slice.h"
#include "exception.h"

SliceNode::SliceNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void SliceNode::create_node()
{
    if(_node)
        return;

    std::vector<float> anchors(_batch_size * _num_of_dims, 0);
    std::vector<float> shape(_batch_size * _num_of_dims, 0);
    std::vector<float> fill_value(_batch_size * _num_of_dims, 0);

    vx_status status;
    _fill_values_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * _num_of_dims);

    status = vxAddArrayItems(_fill_values_array, _batch_size * _num_of_dims, anchors.data(), sizeof(vx_float32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtrppNode_Slice) node: "+ TOSTR(status));

    vx_scalar normalized_anchor = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_anchor);
    vx_scalar normalized_shape = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalized_shape);
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    vx_scalar axis_mask = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);


    _node = vxExtrppNode_Slice(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _anchor->handle(),
                                _shape->handle(), _fill_values_array, axis_mask, normalized_anchor , normalized_shape, policy, _batch_size);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Slice) node failed: "+ TOSTR(status))

}

void SliceNode::update_node()
{
    std::cerr<<"\n SliceNode::update_node()";
    vx_status src_roi_status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size * 4, sizeof(vx_uint32), _inputs[0]->info().get_roi()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vx_status anchor_status  =  (vxQueryTensor((vx_tensor)_anchor->handle(), VX_TENSOR_BUFFER_HOST, &_anchor_array, sizeof(vx_float32)));
    vx_status shape_status = (vxQueryTensor((vx_tensor)_shape->handle(), VX_TENSOR_BUFFER_HOST, &_shape_array, sizeof(vx_float32)));


    if((src_roi_status || anchor_status || shape_status) != 0)
        THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src_roi_status))
    auto audio_roi = _inputs[0]->info().get_roi();
    // The Audio Data is always assumed to be 2d (Keeping 2nd dim as "1" if its a 1d data)
    // Hence introducing the variable num_of_dims_shapes_anchors which can just be a 1d data from NSR
    int num_of_dims_shapes_anchors;
    if (_shape_array[_batch_size] <= 0)
        num_of_dims_shapes_anchors = 1;
    else if ((_shape_array[_batch_size*2] > 0))
        num_of_dims_shapes_anchors = 3;
    else
        num_of_dims_shapes_anchors = 2;
    for(unsigned i = 0; i < _batch_size; i++) {
        int idx = i * num_of_dims_shapes_anchors;
        //TODO: Swetha : To clean up the debug code
            std::cerr << "\n roi x1 " << audio_roi->at(i).x1;
            std::cerr << "\n roi y1 " << audio_roi->at(i).y1;
        for(unsigned d = 0; d < num_of_dims_shapes_anchors; d++) {
        std::cerr << "\n Anchor : " << _anchor_array[idx + d] << "|\t Shape Array : " << (_shape_array[idx + d] - _anchor_array[idx + d]);
        //TODO: Swetha : To handle 3d data by checking NCHW / NHWC format for images
            if(_shape_array[i + _batch_size] > 0  ) { // 2d anchors & shapes
                if (d==0) _output_width_vector[i] = (_shape_array[idx + d] - _anchor_array[idx + d]);
                if (d==1) _output_height_vector[i] = (_shape_array[idx + d] - _anchor_array[idx + d]);
            }
            else if (_shape_array[i] > 0 && num_of_dims_shapes_anchors == 1) { // 1d anchors & shapes
                _output_width_vector[i] = (_shape_array[i] - _anchor_array[i]);
                _output_height_vector[i] = audio_roi->at(i).y1;
            }
            else {
                _output_width_vector[i] = audio_roi->at(i).x1;
                _output_height_vector[i] = audio_roi->at(i).y1;
            }
            _fill_values_vec[idx + d] = _fill_values[0];

            std::cerr << "\n Slice.cpp" << _output_width_vector[i] << " : " << _output_height_vector[i] << " : " << _fill_values_vec[idx + d] << "\t";
        }

        
        std::cerr << "\n";
    }
    
    vx_status status = VX_SUCCESS;
    status |= vxCopyArrayRange((vx_array)_fill_values_array, 0, _batch_size * _num_of_dims, sizeof(vx_float32), _fill_values_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != 0)
        WRN("ERROR: vxCopyArrayRange failed in the normalize node (vxExtrppNode_Slice)  node: "+ TOSTR(status))

    _fill_values_vec.clear();

    _outputs[0]->update_tensor_roi(_output_width_vector, _output_height_vector);
    _anchor_vec.clear();
    _shape_vec.clear();
}

void SliceNode::init(RocalTensor anchor, RocalTensor shape, std::vector<float> &fill_values, std::vector<unsigned> &axes, bool normalized_anchor, bool normalized_shape, RocalOutOfBoundsPolicy policy)
{
    _normalized_anchor = normalized_anchor;
    _normalized_shape = normalized_shape;
    _policy = policy;
    _num_of_dims = _inputs[0]->info().num_of_dims() - 1;
    _anchor = anchor;
    _shape = shape;
    _fill_values = fill_values;
    for(int d = 0; d < axes.size(); d++)
        _axis_mask |= (1 << axes[d]);
    _anchor_vec.resize(_batch_size * _num_of_dims);
    _shape_vec.resize(_batch_size * _num_of_dims);
    _output_width_vector.resize(_batch_size);
    _output_height_vector.resize(_batch_size);
    _fill_values_vec.resize(_batch_size * _num_of_dims);
}