#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"


WarpAffineNode::WarpAffineNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        Node(inputs, outputs),
        _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1])
{
}

void WarpAffineNode::create_node()
{
    std::cerr<<"\n\nWarpAffineNode::create_node()\n\n";
    if(_node)
        return;

    vx_status width_status, height_status;
    _affine.resize(6 * _batch_size);

    uint batch_size = _batch_size;
    for (uint i=0; i < batch_size; i++ )
    {
         _affine[i*6 + 0] = _x0.renew();
         _affine[i*6 + 1] = _y0.renew();
         _affine[i*6 + 2] = _x1.renew();
         _affine[i*6 + 3] = _y1.renew();
         _affine[i*6 + 4] = _o0.renew();
         _affine[i*6 + 5] = _o1.renew();

    }
    // _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    // _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    // std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    // std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());
    // width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    // height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    // if(width_status != 0 || height_status != 0)
    //     THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_WarpAffinePD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);
    // _affine_array.create_array(_graph , VX_TYPE_FLOAT32, _batch_size*6);

    status = vxAddArrayItems(_affine_array,_batch_size * 6, _affine.data(), sizeof(vx_float32));
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    vx_scalar interpolation = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_interpolation_type);
    std::cerr<<"check 1111\n\n";
    _node = vxExtrppNode_WarpAffine (_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _affine_array,interpolation, layout, roi_type, _batch_size);

    std::cerr<<"check2222\n\n";
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffinePD) node failed: "+ TOSTR(status))
}

void WarpAffineNode::update_affine_array()
{
    for (uint i = 0; i < _batch_size; i++ )
    {
        _affine[i*6 + 0] = _x0.renew();
        _affine[i*6 + 1] = _y0.renew();
        _affine[i*6 + 2] = _x1.renew();
        _affine[i*6 + 3] = _y1.renew();
        _affine[i*6 + 4] = _o0.renew();
        _affine[i*6 + 5] = _o1.renew();
    }
    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    if(affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtrppNode_WarpAffinePD) node: "+ TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1,int interpolation_type, int layout )
{
    std::cerr<<"WarpAffineNode::init\n\n";
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
    _interpolation_type=interpolation_type;
}

void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1,int interpolation_type, int layout)
{
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
    _interpolation_type=interpolation_type;
}

void WarpAffineNode::update_node()
{
    update_affine_array();
}
