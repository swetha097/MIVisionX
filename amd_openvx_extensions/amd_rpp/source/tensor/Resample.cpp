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

#include "internal_publishKernels.h"
#include "vx_ext_amd.h"

struct ResampleLocalData 
{
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    float quality;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    Rpp32s *sampleFrames;
    Rpp32s *sampleChannels;
    Rpp32f *inRateTensor;
    Rpp32f *outRateTensor;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

void update_destination_roi(const vx_reference *parameters, ResampleLocalData *data, RpptROI* src_roi, RpptROI* dst_roi) {
    float _scale_ratio;
    for(unsigned i = 0; i < data->pSrcDesc->n; i++) {
        if (data->inRateTensor[i] != 0)
            _scale_ratio = data->outRateTensor[i] / (float)data->inRateTensor[i];
        else
            _scale_ratio = 0;
        dst_roi[i].xywhROI.xy.x = (int)std::ceil(_scale_ratio * src_roi[i].xywhROI.xy.x);
        dst_roi[i].xywhROI.xy.y = src_roi[i].xywhROI.xy.y;
    }
}

static vx_status VX_CALLBACK refreshResample(vx_node node, const vx_reference *parameters, vx_uint32 num, ResampleLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->pSrcDesc->n, sizeof(float), data->inRateTensor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->outRateTensor, sizeof(data->outRateTensor)));
    void *roi_tensor_ptr_src, *roi_tensor_ptr_dst;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));

    }
    RpptROI* src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_src);
    RpptROI* dst_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_dst);
    for (uint i = 0; i < data->pSrcDesc->n ; i++) {
        data->sampleFrames[i] = src_roi[i].xywhROI.xy.x;
        data->sampleChannels[i] = src_roi[i].xywhROI.xy.y;
    }
    update_destination_roi(parameters, data, src_roi, dst_roi);
    return status;
}

static vx_status VX_CALLBACK validateResample(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;

    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 1);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ResampleLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        refreshResample(node, parameters, num, data);
        rpp_status = rppt_resample_host((float *)data->pSrc, data->pSrcDesc, (float *)data->pDst, data->pDstDesc,
                                       (float *)data->inRateTensor,(float *)data->outRateTensor, data->sampleFrames, data->sampleChannels,
                                        data->quality, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResampleLocalData *data = new ResampleLocalData;
    memset(data, 0, sizeof(ResampleLocalData));

    vx_enum input_tensor_dtype, output_tensor_dtype;

    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->quality));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;
    fillAudioDescriptionPtrFromDims(data->pSrcDesc, data->inputTensorDims);

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1],VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillAudioDescriptionPtrFromDims(data->pDstDesc, data->outputTensorDims);

    data->sampleFrames = static_cast<signed int *>(calloc(data->pSrcDesc->n, sizeof(signed int)));
    data->sampleChannels = static_cast<signed int *>(calloc(data->pSrcDesc->n, sizeof(signed int)));
    data->inRateTensor = static_cast<float *>(calloc(data->pSrcDesc->n, sizeof(float)));
    refreshResample(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResampleLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    if(data->sampleFrames != nullptr) free(data->sampleFrames);
    if(data->sampleChannels != nullptr) free(data->sampleChannels);
    if(data->inRateTensor != nullptr) free(data->inRateTensor);
    delete(data->pSrcDesc);
    delete(data->pDstDesc);
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status Resample_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Resample",
                                       VX_KERNEL_RPP_RESAMPLE,
                                       processResample,
                                       8,
                                       validateResample,
                                       initializeResample,
                                       uninitializeResample);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        // PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // maxDstWidth
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
