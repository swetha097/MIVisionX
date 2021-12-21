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

#include "internal_publishKernels.h"

struct RicapLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppiSize *srcDimensions;
    RppiSize maxSrcDimensions;
    Rpp32u *srcBatch_width;
    Rpp32u *srcBatch_height;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *permutedIndices1;
    vx_uint32 *permutedIndices2;
    vx_uint32 *permutedIndices3;
    vx_uint32 *permutedIndices4;
    vx_uint32 *cropCoords1;
    vx_uint32 *cropCoords2;
    vx_uint32 *cropCoords3;
    vx_uint32 *cropCoords4;
    RpptDescPtr srcDescPtr, dstDescPtr;
    RpptROIPtr roiTensorPtrSrc;
    RpptRoiType roiType;
    RpptROI roiTensorSrc;
    RpptDesc srcDesc, dstDesc;
    RpptROI *roiPtrInputCropRegion;
    Rpp32u *permutedArrayOrderChanged;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK refreshRicap(vx_node node, const vx_reference *parameters, vx_uint32 num, RicapLocalData *data)
{
    vx_status status = VX_SUCCESS;
    vx_status copy_status;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_uint32), data->permutedIndices1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(vx_uint32), data->permutedIndices2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->nbatchSize, sizeof(vx_uint32), data->permutedIndices3, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[7], 0, data->nbatchSize, sizeof(vx_uint32), data->permutedIndices4, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[8], 0, 4, sizeof(vx_uint32), data->cropCoords1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[9], 0, 4, sizeof(vx_uint32), data->cropCoords2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[10], 0, 4, sizeof(vx_uint32), data->cropCoords3, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[11], 0, 4, sizeof(vx_uint32), data->cropCoords4, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->maxSrcDimensions.height, sizeof(data->maxSrcDimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->maxSrcDimensions.width, sizeof(data->maxSrcDimensions.width)));
    data->maxSrcDimensions.height = data->maxSrcDimensions.height / data->nbatchSize;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->nbatchSize; i++)
    {
        data->srcDimensions[i].width = data->roiTensorPtrSrc[i].xywhROI.roiWidth = data->srcBatch_width[i];
        data->srcDimensions[i].height = data->roiTensorPtrSrc[i].xywhROI.roiHeight = data->srcBatch_height[i];
        data->roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        data->roiTensorPtrSrc[i].xywhROI.xy.y = 0;
    }
    // xywhROI override sample cropCoordsROI
    data->roiPtrInputCropRegion[0].xywhROI.xy.x = data->cropCoords1[0];
    data->roiPtrInputCropRegion[0].xywhROI.xy.y = data->cropCoords1[1];
    data->roiPtrInputCropRegion[0].xywhROI.roiWidth = data->cropCoords1[2];
    data->roiPtrInputCropRegion[0].xywhROI.roiHeight = data->cropCoords1[3];

    data->roiPtrInputCropRegion[1].xywhROI.xy.x = data->cropCoords2[0];
    data->roiPtrInputCropRegion[1].xywhROI.xy.y = data->cropCoords2[1];
    data->roiPtrInputCropRegion[1].xywhROI.roiWidth = data->cropCoords2[2];
    data->roiPtrInputCropRegion[1].xywhROI.roiHeight = data->cropCoords2[3];

    data->roiPtrInputCropRegion[2].xywhROI.xy.x = data->cropCoords3[0];
    data->roiPtrInputCropRegion[2].xywhROI.xy.y = data->cropCoords3[1];
    data->roiPtrInputCropRegion[2].xywhROI.roiWidth = data->cropCoords3[2];
    data->roiPtrInputCropRegion[2].xywhROI.roiHeight = data->cropCoords3[3];

    data->roiPtrInputCropRegion[3].xywhROI.xy.x = data->cropCoords4[0];
    data->roiPtrInputCropRegion[3].xywhROI.xy.y = data->cropCoords4[1];
    data->roiPtrInputCropRegion[3].xywhROI.roiWidth = data->cropCoords4[2];
    data->roiPtrInputCropRegion[3].xywhROI.roiHeight = data->cropCoords4[3];

    // Permuted Indices Order changed
    for (uint i = 0, j = 0; i < data->nbatchSize, j < data->nbatchSize * 4; i++, j += 4)
    {
        data->permutedArrayOrderChanged[j] = data->permutedIndices1[i];
        data->permutedArrayOrderChanged[j + 1] = data->permutedIndices2[i];
        data->permutedArrayOrderChanged[j + 2] = data->permutedIndices3[i];
        data->permutedArrayOrderChanged[j + 3] = data->permutedIndices4[i];
    }
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pDst, sizeof(data->hip_pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc, sizeof(vx_uint8)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pDst, sizeof(vx_uint8)));
    }
    return status;
}

static vx_status VX_CALLBACK validateRicap(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[12], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #12 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[13], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #13 type=%d (must be size)\n", scalar_type);
    // Check for input parameters
    vx_parameter input_param;
    vx_image input;
    vx_df_image df_image;
    input_param = vxGetParameterByIndex(node, 0);
    STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_image)));
    STATUS_ERROR_CHECK(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB)
    {
        return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: Ricap: image: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&df_image);
    }

    // Check for output parameters
    vx_image output;
    vx_df_image format;
    vx_parameter output_param;
    vx_uint32 height, width;
    output_param = vxGetParameterByIndex(node, 3);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseParameter(&output_param);
    vxReleaseParameter(&input_param);
    return status;
}

static vx_status VX_CALLBACK processRicap(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    RicapLocalData *data = NULL;
    vx_int32 output_format_toggle = 0;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
//     if (data->device_type == AGO_TARGET_AFFINITY_GPU) // Support not yet given in RPP for gpu kernels
//     {
// #if ENABLE_OPENCL
//         refreshRicap(node, parameters, num, data);
//         if (df_image == VX_DF_IMAGE_U8)
//         {
//             rpp_status = rppi_ricap_u8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
//         }
//         else if (df_image == VX_DF_IMAGE_RGB)
//         {
//             rpp_status = rppi_ricap_u8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
//         }
//         return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
// #elif ENABLE_HIP
//         refreshRicap(node, parameters, num, data);
//         if (df_image == VX_DF_IMAGE_U8)
//         {
//             rpp_status = rppi_ricap_u8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
//         }
//         else if (df_image == VX_DF_IMAGE_RGB)
//         {
//             rpp_status = rppi_ricap_u8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
//         }
//         return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
// #endif
//     }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshRicap(node, parameters, num, data);
        if (df_image == VX_DF_IMAGE_U8)
        {
            // rpp_status = rppi_ricap_u8_pln1_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
            rpp_status = rppt_ricap_host(data->pSrc, data->srcDescPtr, data->pDst, data->dstDescPtr, data->permutedArrayOrderChanged, data->roiPtrInputCropRegion, data->roiTensorPtrSrc, data->roiType, data->rppHandle);

        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            rpp_status = rppt_ricap_host(data->pSrc, data->srcDescPtr, data->pDst, data->dstDescPtr, data->permutedArrayOrderChanged, data->roiPtrInputCropRegion, data->roiTensorPtrSrc, data->roiType, data->rppHandle);

            // rpp_status = rppi_ricap_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->permutedIndices1, data->permutedIndices2, data->permutedIndices3, data->permutedIndices4, data->cropCoords1, data->cropCoords2, data->cropCoords3, data->cropCoords4, output_format_toggle, data->nbatchSize, data->rppHandle);
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeRicap(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RicapLocalData *data = new RicapLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[13], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[12], &data->nbatchSize));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->maxSrcDimensions.height, sizeof(data->maxSrcDimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->maxSrcDimensions.width, sizeof(data->maxSrcDimensions.width)));
    data->maxSrcDimensions.height = data->maxSrcDimensions.height / data->nbatchSize;
    data->permutedIndices1 = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->permutedIndices2 = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->permutedIndices3 = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->permutedIndices4 = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->cropCoords1 = (vx_uint32 *)malloc(sizeof(vx_uint32) * 4);
    data->cropCoords2 = (vx_uint32 *)malloc(sizeof(vx_uint32) * 4);
    data->cropCoords3 = (vx_uint32 *)malloc(sizeof(vx_uint32) * 4);
    data->cropCoords4 = (vx_uint32 *)malloc(sizeof(vx_uint32) * 4);
    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->nbatchSize);
    data->srcBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->srcBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->permutedArrayOrderChanged = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize * 4);

    // Initializing tensor config parameters.

    uint ip_channel = 3;
    data->srcDescPtr = &data->srcDesc;
    data->dstDescPtr = &data->dstDesc;
    data->srcDescPtr->layout = RpptLayout::NHWC;
    data->dstDescPtr->layout = RpptLayout::NHWC;

    data->srcDescPtr->dataType = RpptDataType::U8;
    data->dstDescPtr->dataType = RpptDataType::U8;

    // Set numDims, offset, n/c/h/w values for src/dst
    data->srcDescPtr->numDims = 4;
    data->dstDescPtr->numDims = 4;

    data->srcDescPtr->offsetInBytes = 0;
    data->dstDescPtr->offsetInBytes = 0;

    data->srcDescPtr->n = data->nbatchSize;
    data->srcDescPtr->h = data->maxSrcDimensions.height;
    data->srcDescPtr->w = data->maxSrcDimensions.width;
    data->srcDescPtr->c = ip_channel;

    data->dstDescPtr->n = data->nbatchSize;
    data->dstDescPtr->h = data->maxSrcDimensions.height;
    data->dstDescPtr->w = data->maxSrcDimensions.width;
    data->dstDescPtr->c = ip_channel;

    // Set n/c/h/w strides for src/dst

    data->srcDescPtr->strides.nStride = ip_channel * data->srcDescPtr->w * data->srcDescPtr->h;
    data->srcDescPtr->strides.hStride = ip_channel * data->srcDescPtr->w;
    data->srcDescPtr->strides.wStride = ip_channel;
    data->srcDescPtr->strides.cStride = 1;

    data->dstDescPtr->strides.nStride = ip_channel * data->dstDescPtr->w * data->dstDescPtr->h;
    data->dstDescPtr->strides.hStride = ip_channel * data->dstDescPtr->w;
    data->dstDescPtr->strides.wStride = ip_channel;
    data->dstDescPtr->strides.cStride = 1;

    // Initialize ROI tensors for src/dst
    data->roiTensorPtrSrc  = (RpptROI *) calloc(data->nbatchSize, sizeof(RpptROI));

    // Set ROI tensors types for src/dst
    data->roiType = RpptRoiType::XYWH;

    data->roiPtrInputCropRegion = (RpptROI *)calloc(4, sizeof(RpptROI));

    refreshRicap(node, parameters, num, data);
#if ENABLE_OPENCL
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.cmdq, data->nbatchSize);
#elif ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.hipstream, data->nbatchSize);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeRicap(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RicapLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->srcBatch_height);
    free(data->srcBatch_width);
    free(data->srcDimensions);
    free(data->permutedIndices1);
    free(data->permutedIndices2);
    free(data->permutedIndices3);
    free(data->permutedIndices4);
    free(data->cropCoords1);
    free(data->cropCoords2);
    free(data->cropCoords3);
    free(data->cropCoords4);
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif
    return VX_SUCCESS;
}

vx_status Ricap_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Ricap",
                                       VX_KERNEL_RPP_RICAP,
                                       processRicap,
                                       14,
                                       validateRicap,
                                       initializeRicap,
                                       uninitializeRicap);
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

    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 13, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
