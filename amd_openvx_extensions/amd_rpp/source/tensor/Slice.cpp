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

struct SliceLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32f *pAnchor;
    Rpp32f *pShape;
    Rpp32f *pFillValues;
    Rpp32s axes;
    bool normalizedAnchor;
    bool normalizedShape;
    Rpp32u policy;
    Rpp32s *pSrcDims;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
    Rpp32u dimsStride;
};

void copy_src_dims_and_update_dst_roi(SliceLocalData *data, RpptROI *src_roi, RpptROI *dst_roi) {
    // Fill the values from ROI buffers to RPP input buffers
    for (unsigned i = 0, j = 0; i < data->inputTensorDims[0]; i++, j += 2) {
        data->pSrcDims[j] = src_roi[i].xywhROI.xy.x;
        data->pSrcDims[j + 1] = src_roi[i].xywhROI.xy.y;
    }
    // if input is 2D - shape will be of size (batchSize * 2) - so fill dst_roi using both values
    if (data->dimsStride == 2) {
        for (unsigned i = 0, j = 0; i < data->inputTensorDims[0]; i++, j += 2) {
            dst_roi[i].xywhROI.xy.x = (data->pShape[j]);
            dst_roi[i].xywhROI.xy.y = (data->pShape[j + 1]);
        }
    } else if (data->dimsStride == 1) {
        // if input is 1D - shape will be of size (batchSize) - so fill only width of dst_roi from buffer and set height to 1
        for (unsigned i = 0; i < data->inputTensorDims[0]; i++) {
            dst_roi[i].xywhROI.xy.x = data->pShape[i];
            dst_roi[i].xywhROI.xy.y = 1;
        }
    }
}

static vx_status VX_CALLBACK refreshSlice(vx_node node, const vx_reference *parameters, vx_uint32 num, SliceLocalData *data) {
    vx_status status = VX_SUCCESS;
    void *roi_tensor_ptr_src, *roi_tensor_ptr_dst;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->pAnchor, sizeof(data->pAnchor)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HOST, &data->pShape, sizeof(data->pShape)));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->pSrcDesc->n * data->dimsStride, sizeof(float), data->pFillValues, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(&data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(&data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
    }

    RpptROI *src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_src);
    RpptROI *dst_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_dst);
    copy_src_dims_and_update_dst_roi(data, src_roi, dst_roi);
    return status;
}

static vx_status VX_CALLBACK validateSlice(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_BOOL)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_BOOL)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #9 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #10 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #11 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Slice: tensor: #0 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    // Check for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Slice: tensor: #2 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    SliceLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshSlice(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return_status = VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        return_status = VX_ERROR_NOT_IMPLEMENTED;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        rpp_status = rppt_slice_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->pSrcDims, data->pAnchor, data->pShape, data->pFillValues, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data = new SliceLocalData;
    memset(data, 0, sizeof(SliceLocalData));

    vx_enum input_tensor_dtype, output_tensor_dtype;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->axes));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[8], &data->normalizedAnchor));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[9], &data->normalizedShape));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[10], &data->policy));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &data->dimsStride));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[12], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

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
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2],VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillAudioDescriptionPtrFromDims(data->pDstDesc, data->outputTensorDims);

    data->pSrcDims = static_cast<int *>(calloc(data->pSrcDesc->n * 2, sizeof(int)));
    data->pFillValues = static_cast<float *>(calloc(data->pSrcDesc->n * data->dimsStride, sizeof(float)));

    refreshSlice(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    free(data->pSrcDims);
    free(data->pFillValues);
    delete(data->pSrcDesc);
    delete(data->pDstDesc);
    delete(data);
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

vx_status Slice_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Slice",
                                       VX_KERNEL_RPP_SLICE,
                                       processSlice,
                                       13,
                                       validateSlice,
                                       initializeSlice,
                                       uninitializeSlice);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}