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
#include <graph.h>
#include "node_resize_mirror_normalize.h"
#include "exception.h"
ResizeMirrorNormalizeTensorNode::ResizeMirrorNormalizeTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])

{
}
    
void ResizeMirrorNormalizeTensorNode::create_node()
{
    std::cerr<<"create_node checking \n";
    if(_node)
        return;
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().get_width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().get_height());
    _mean_vx.resize(_batch_size*3);
    _std_dev_vx.resize(_batch_size*3);
    for (uint i=0; i < _batch_size; i++ ) {
        _mean_vx[3*i] = _mean[0];
        _mean_vx[3*i+1] = _mean[1];
        _mean_vx[3*i+2] = _mean[2];

        _std_dev_vx[3*i] = _std_dev[0];
        _std_dev_vx[3*i+1] = _std_dev[1];
        _std_dev_vx[3*i+2] = _std_dev[2];
    }
std::cerr<<"create_node checking 222222\n";
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
   
    
    vx_status status = VX_SUCCESS;
    std::cerr<<"create_node checking 333\n";
    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size*3);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size*3);

    status |= vxAddArrayItems(_mean_array,_batch_size*3, _mean_vx.data(), sizeof(vx_float32));
    std::cerr<<"create_node checking 44444\n";

    status |= vxAddArrayItems(_std_dev_array,_batch_size*3, _std_dev_vx.data(), sizeof(vx_float32));
    std::cerr<<"create_node checking 5555\n";

    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    vx_status width_status, height_status;
std::cerr<<"create_node checking 6666\n";

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    std::cerr<<"dst_roi_height "<<dst_roi_height[0]<<"  "<<dst_roi_width[0];
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));
    bool packed;
    vx_scalar interpolation = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_interpolation_type);
std::cerr<<"create_node checking 44444\n";

    unsigned int chnShift = 0;
    vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
    vx_scalar is_packed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_BOOL,&packed);

    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    std::cerr<<"node checking 11111\n";
   _node = vxExtrppNode_ResizeMirrorNormalize(_graph->get(), _inputs[0]->handle(),
                                                   _src_tensor_roi,_outputs[0]->handle(),_src_tensor_roi,_dst_roi_width,_dst_roi_height,
                                                   interpolation,_mean_array, _std_dev_array, _mirror.default_array() ,
                                                   is_packed, chnToggle,layout, roi_type, _batch_size);
    std::cerr<<"node checking 22222222\n";
    // vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))

}
void ResizeMirrorNormalizeTensorNode::update_node()
{
    _mirror.update_array();

}
void ResizeMirrorNormalizeTensorNode::init(int interpolation_type,std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror, int layout)
{
    std::cerr<<"init checking \n";
  _interpolation_type=interpolation_type;
  _mean   = mean;
  _std_dev = std_dev;
  _mirror.set_param(core(mirror));
  _layout=layout;
  
}
