#include <vx_ext_rpp.h>
#include "node_spatter.h"
#include "exception.h"

SpatterTensorNode::SpatterTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs)


{
}

void SpatterTensorNode::create_node()
{
    if(_node)
        return;

    
    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar red_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_red);
    vx_scalar green_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_green);
    vx_scalar blue_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_blue);

    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_Spatter(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), red_val, green_val, blue_val, layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the spatter_batch (vxExtrppNode_spatter) node failed: "+ TOSTR(status))
}

void SpatterTensorNode::init( int red, int green, int blue,int layout)
{
    _layout=layout;
    _red = red;
    _green = green;
    _blue = blue;
}




void SpatterTensorNode::update_node()
{
    
}
