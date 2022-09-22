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

#include "internal_publishKernels.h"

struct CropMirrorNormalizetensorLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppiSize *srcDimensions;
    RppiSize maxSrcDimensions;
    Rpp32u *srcBatch_width;
    Rpp32u *srcBatch_height;
    RppiSize *dstDimensions;
    RppiSize maxDstDimensions;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *start_x;
    vx_uint32 *start_y;
    vx_float32 *mean;
    vx_float32 *std_dev;
    vx_uint32 *mirror;
    vx_uint32 chnShift; //NHWC to NCHW
    Rpp32u *dstBatch_width;
    Rpp32u *dstBatch_height;
    RpptROIPtr roiTensorPtrSrc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    RpptDesc srcDesc, dstDesc;
    RpptRoiType roiType;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK refreshCropMirrorNormalizetensor(vx_node node, const vx_reference *parameters, vx_uint32 num, CropMirrorNormalizetensorLocalData *data)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->nbatchSize, sizeof(vx_uint32), data->start_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[7], 0, data->nbatchSize, sizeof(vx_uint32), data->start_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[8], 0, data->nbatchSize, sizeof(vx_float32), data->mean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[9], 0, data->nbatchSize, sizeof(vx_float32), data->std_dev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[10], 0, data->nbatchSize, sizeof(vx_uint32), data->mirror, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &data->chnShift));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->maxSrcDimensions.height, sizeof(data->maxSrcDimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->maxSrcDimensions.width, sizeof(data->maxSrcDimensions.width)));
    data->maxSrcDimensions.height = data->maxSrcDimensions.height / data->nbatchSize;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_HEIGHT, &data->maxDstDimensions.height, sizeof(data->maxDstDimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_WIDTH, &data->maxDstDimensions.width, sizeof(data->maxDstDimensions.width)));
    data->maxDstDimensions.height = data->maxDstDimensions.height / data->nbatchSize;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(Rpp32u), data->dstBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(Rpp32u), data->dstBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->nbatchSize; i++)
    {
        data->srcDimensions[i].width = data->srcBatch_width[i];
        data->srcDimensions[i].height = data->srcBatch_height[i];
        data->dstDimensions[i].width = data->dstBatch_width[i];
        data->dstDimensions[i].height = data->dstBatch_height[i];
        data->roiTensorPtrSrc[i].xywhROI.xy.x = data->start_x[i];
        data->roiTensorPtrSrc[i].xywhROI.xy.y =data->start_y[i];
        data->roiTensorPtrSrc[i].xywhROI.roiHeight =data->dstBatch_width[i];
        data->roiTensorPtrSrc[i].xywhROI.roiWidth=data->dstBatch_height[i];


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

static vx_status VX_CALLBACK validateCropMirrorNormalizetensor(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #11 type=%d (must be size)\n", scalar_type);
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
        return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: CropMirrorNormalizebatchPD: image: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&df_image);
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

static vx_status VX_CALLBACK processCropMirrorNormalizetensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    CropMirrorNormalizetensorLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        refreshCropMirrorNormalizetensor(node, parameters, num, data);
        if (df_image == VX_DF_IMAGE_U8)
        {
            // rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            // rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        refreshCropMirrorNormalizetensor(node, parameters, num, data);
        if (df_image == VX_DF_IMAGE_U8)
        {
            // rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            // rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshCropMirrorNormalizetensor(node, parameters, num, data);
        // if (df_image == VX_DF_IMAGE_U8)
        // {
        //     rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        // }
        // else if (df_image == VX_DF_IMAGE_RGB)
        // {
        //     rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions, data->start_x, data->start_y, data->mean, data->std_dev, data->mirror, data->chnShift, data->nbatchSize, data->rppHandle);
        // }
        if (1) {
            float *temp1 = ((float *)calloc(100, sizeof(float)));
            for (int i = 0; i < 100; i++) {
                temp1[i] = (float)*((unsigned char *)(data->pSrc) + i);
                std::cout << temp1[i] << " ";
            }
        }
        std::cerr<<"\n";
        std::cerr<<"\n\nProcess check3"<<data->mean[0]<<" "<<data->std_dev[0] <<" "<<data-> mirror[0]<<"\n";
        rpp_status = rppt_crop_mirror_normalize_host(data->pSrc, data->srcDescPtr, data->pDst, data->dstDescPtr, data->mean, data->std_dev,data-> mirror, data->roiTensorPtrSrc, data->roiType, data->rppHandle);
        if (1) {
            float *temp1 = ((float *)calloc(100, sizeof(float)));
            for (int i = 0; i < 100; i++) {
                temp1[i] = (float)*((unsigned char *)(data->pDst) + i);
                std::cout << temp1[i] << " ";
            }
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeCropMirrorNormalizetensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    CropMirrorNormalizetensorLocalData *data = new CropMirrorNormalizetensorLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[13], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[12], &data->nbatchSize));
    data->start_x = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->start_y = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->mean = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize);
    data->std_dev = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize);
    data->mirror = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->nbatchSize);
    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->nbatchSize);
    data->dstDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->nbatchSize);
    data->srcBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->srcBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->dstBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->dstBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    
    std::cerr<<"check 11 in initialization\n\n";
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->maxSrcDimensions.height, sizeof(data->maxSrcDimensions.height)));
    std::cerr<<"data->maxSrcDimensions.height"<<data->maxSrcDimensions.height<<"\n";
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->maxSrcDimensions.width, sizeof(data->maxSrcDimensions.width)));
    std::cerr<<"data->maxSrcDimensions.width "<<data->maxSrcDimensions.width<<"\n";

    data->maxSrcDimensions.height = data->maxSrcDimensions.height / data->nbatchSize;
    std::cerr<<"data->maxSrcDimensions.height "<<data->maxSrcDimensions.height<<"\n";

    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_HEIGHT, &data->maxDstDimensions.height, sizeof(data->maxDstDimensions.height)));
    std::cerr<<"data->maxDstDimensions.height"<<data->maxDstDimensions.height<<"\n";

    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_WIDTH, &data->maxDstDimensions.width, sizeof(data->maxDstDimensions.width)));
    std::cerr<<"data->maxDstDimensions.width "<<data->maxDstDimensions.width<<"\n";

    data->maxDstDimensions.height = data->maxDstDimensions.height / data->nbatchSize;
    std::cerr<<"data->maxDstDimensions.height "<<data->maxDstDimensions.height<<"\n";
    
    // Check if it is a RGB or single channel U8 input
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    uint ip_channel = (df_image == VX_DF_IMAGE_RGB) ? 3 : 1;


    data->roiTensorPtrSrc = (RpptROI *) calloc(data->nbatchSize, sizeof(RpptROI));
    data->roiType = RpptRoiType::XYWH;
    data->srcDescPtr = &data->srcDesc;
    data->dstDescPtr = &data->dstDesc;

    data->srcDescPtr->dataType = RpptDataType::U8;
    data->dstDescPtr->dataType = RpptDataType::U8;

    data->srcDescPtr->numDims = 4;
    data->dstDescPtr->numDims = 4;
    data->srcDescPtr->offsetInBytes = 0;
    data->dstDescPtr->offsetInBytes = 0;
    data->srcDescPtr->n = data->nbatchSize;
    data->srcDescPtr->h = data->maxSrcDimensions.height;
    data->srcDescPtr->w = data->maxSrcDimensions.width;
    data->srcDescPtr->c = ip_channel;
    data->dstDescPtr->n = data->nbatchSize;
    data->dstDescPtr->h = data->maxDstDimensions.height;
    data->dstDescPtr->w = data->maxDstDimensions.width;
    data->dstDescPtr->c = ip_channel;

        if(df_image == VX_DF_IMAGE_U8) // For PLN1 images
    {
        data->srcDescPtr->layout = RpptLayout::NCHW;
        data->dstDescPtr->layout = RpptLayout::NCHW;
        data->srcDescPtr->strides.nStride = ip_channel * data->srcDescPtr->w * data->srcDescPtr->h;
        data->srcDescPtr->strides.cStride = data->srcDescPtr->w * data->srcDescPtr->h;
        data->srcDescPtr->strides.hStride = data->srcDescPtr->w;
        data->srcDescPtr->strides.wStride = 1;
        data->dstDescPtr->strides.nStride = ip_channel * data->dstDescPtr->w * data->dstDescPtr->h;
        data->dstDescPtr->strides.cStride = data->dstDescPtr->w * data->dstDescPtr->h;
        data->dstDescPtr->strides.hStride = data->dstDescPtr->w;
        data->dstDescPtr->strides.wStride = 1;
    }
    else // For RGB (NHWC/NCHW) images
    {
        data->srcDescPtr->layout = RpptLayout::NHWC;
        data->dstDescPtr->layout = RpptLayout::NHWC;
        data->srcDescPtr->strides.nStride = ip_channel * data->srcDescPtr->w * data->srcDescPtr->h;
        data->srcDescPtr->strides.hStride = ip_channel * data->srcDescPtr->w;
        data->srcDescPtr->strides.wStride = ip_channel;
        data->srcDescPtr->strides.cStride = 1;
        data->dstDescPtr->strides.nStride = ip_channel * data->dstDescPtr->w * data->dstDescPtr->h;
        data->dstDescPtr->strides.hStride = ip_channel * data->dstDescPtr->w;
        data->dstDescPtr->strides.wStride = ip_channel;
        data->dstDescPtr->strides.cStride = 1;
    }
    refreshCropMirrorNormalizetensor(node, parameters, num, data);
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

static vx_status VX_CALLBACK uninitializeCropMirrorNormalizetensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    CropMirrorNormalizetensorLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->start_x);
    free(data->start_y);
    free(data->mean);
    free(data->std_dev);
    free(data->mirror);
    free(data->srcDimensions);
    free(data->dstDimensions);
    free(data->srcBatch_width);
    free(data->srcBatch_height);
    free(data->dstBatch_width);
    free(data->dstBatch_height);
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

vx_status CropMirrorNormalizetensor_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.CropMirrorNormalizetensor",
                                       VX_KERNEL_RPP_CROPMIRRORNORMALIZETENSOR,
                                       processCropMirrorNormalizetensor,
                                       14,
                                       validateCropMirrorNormalizetensor,
                                       initializeCropMirrorNormalizetensor,
                                       uninitializeCropMirrorNormalizetensor);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
