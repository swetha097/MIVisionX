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

struct NonSilentRegionLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    Rpp32f *pDst1;
    Rpp32f *pDst2;
    Rpp32f cutOffDB;
    Rpp32f referencePower;
    Rpp32s windowLength;
    Rpp32s resetInterval;
    RpptDescPtr pSrcDesc;
    Rpp32s *pSrcLength;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t ouputTensorDims[RPP_MAX_TENSOR_DIMS];
};

static vx_status VX_CALLBACK refreshNonSilentRegion(vx_node node, const vx_reference *parameters, vx_uint32 num, NonSilentRegionLocalData *data) {
    vx_status status = VX_SUCCESS;
    void *roi_tensor_ptr;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst1, sizeof(data->pDst1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->pDst2, sizeof(data->pDst2)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst1, sizeof(data->pDst1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst2, sizeof(data->pDst2)));
    }
    RpptROI *srcRoi = reinterpret_cast<RpptROI *>(roi_tensor_ptr);
    for (unsigned i = 0; i < data->inputTensorDims[0]; i++)
        data->pSrcLength[i] = static_cast<int>(srcRoi[i].xywhROI.xy.x);

    return status;
}

static vx_status VX_CALLBACK validateNonSilentRegion(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: NonSilentRegion: tensor: #0 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    // Check for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: NonSilentRegion: tensor: #2 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: NonSilentRegion: tensor: #3 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processNonSilentRegion(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    NonSilentRegionLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshNonSilentRegion(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return_status = VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        // rpp_status = rppt_non_silent_region_detection_gpu(data->pSrc, pSrcDesc, data->pSrcLength, data->pDst1, data->pDst2, data->cutOffDB, data->windowLength, data->referencePower, data->resetInterval, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        rpp_status = rppt_non_silent_region_detection_host(data->pSrc, data->pSrcDesc, data->pSrcLength, data->pDst1, data->pDst2, data->cutOffDB, data->windowLength, data->referencePower, data->resetInterval, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeNonSilentRegion(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    NonSilentRegionLocalData *data = new NonSilentRegionLocalData;
    memset(data, 0, sizeof(*data));

    vx_enum input_tensor_datatype;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[4], &data->cutOffDB));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[5], &data->referencePower));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->windowLength));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->resetInterval));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_datatype, sizeof(input_tensor_datatype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_datatype);
    data->pSrcDesc->offsetInBytes = 0;

    // fill dims and strides in srcDescPtr
    data->pSrcDesc->n = data->inputTensorDims[0];
    data->pSrcDesc->h = data->inputTensorDims[2];
    data->pSrcDesc->w = data->inputTensorDims[1];
    data->pSrcDesc->c = 1;
    data->pSrcDesc->strides.nStride = data->pSrcDesc->c * data->pSrcDesc->w * data->pSrcDesc->h;
    data->pSrcDesc->strides.hStride = data->pSrcDesc->c * data->pSrcDesc->w;
    data->pSrcDesc->strides.wStride = data->pSrcDesc->c;
    data->pSrcDesc->strides.cStride = 1;

    data->pSrcDesc->numDims = 4;
    data->pSrcLength = static_cast<int *>(calloc(data->pSrcDesc->n, sizeof(int)));
    refreshNonSilentRegion(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeNonSilentRegion(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    NonSilentRegionLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->pSrcLength != nullptr) free(data->pSrcLength);
    delete(data->pSrcDesc);
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
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

vx_status NonSilentRegion_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.NonSilentRegion",
                                       VX_KERNEL_RPP_NONSILENTREGION,
                                       processNonSilentRegion,
                                       9,
                                       validateNonSilentRegion,
                                       initializeNonSilentRegion,
                                       uninitializeNonSilentRegion);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
