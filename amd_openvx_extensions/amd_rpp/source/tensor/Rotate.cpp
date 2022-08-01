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
#define NUM_OF_DIMS 5

struct RotateLocalData {
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    RpptROI *roi_tensor_Ptr;
    RpptRoiType roiType;
    size_t in_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    size_t out_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    Rpp32u layout;
    Rpp32u OutputFormatToggle;
    RppiSize *srcDimensions; // TBR : Not present in tensor
    RppiSize maxSrcDimensions;  // TBR : Not present in tensor
    RppiSize *dstDimensions; // TBR : Not present in tensor
    RppiSize maxDstDimensions;  // TBR : Not present in tensor
    Rpp32u *srcBatch_width; // TBR : Not present in tensor
    Rpp32u *srcBatch_height;    // TBR : Not present in tensor
    Rpp32u *dstBatch_width; // TBR : Not present in tensor
    Rpp32u *dstBatch_height;    // TBR : Not present in tensor
    vx_float32 *kernelSize;
    RpptDescPtr src_desc_ptr;
    RpptDescPtr dst_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK refreshRotate(vx_node node, const vx_reference *parameters, vx_uint32 num, RotateLocalData *data) {
    std::cerr << "refreshRotate\n\n";
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize * 4, sizeof(unsigned), data->roi_tensor_Ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(vx_float32), data->kernelSize, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_uint32), data->dstBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->nbatchSize, sizeof(vx_uint32), data->dstBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    if (data->layout == 0 || data->layout == 1) 
    {
        for (int i = 0; i < data->nbatchSize; i++)
        {
            data->srcDimensions[i].width = data->roi_tensor_Ptr[i].xywhROI.roiWidth;
            data->srcDimensions[i].height = data->roi_tensor_Ptr[i].xywhROI.roiHeight;

            data->dstDimensions[i].width = data->dstBatch_width[i];
            data->dstDimensions[i].height = data->dstBatch_height[i];
std::cerr<<"  *****************************"<<data->dstDimensions[i].width<<"   "<<data->dstDimensions[i].height<<"\n\n"<<data->dstBatch_width[i]<<"  "<<data->dstBatch_height[i]<<"\n";

        }
    }
    if (data->layout == 2 || data->layout == 3) {
        unsigned num_of_frames = data->in_tensor_dims[1];  // Num of frames 'F'
        for (int n = data->nbatchSize - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for (int f = 0; f < num_of_frames; f++) {
                data->kernelSize[index + f] = data->kernelSize[n];
                data->srcDimensions[index + f].width = data->roi_tensor_Ptr[n].xywhROI.roiWidth;
                data->srcDimensions[index + f].height = data->roi_tensor_Ptr[n].xywhROI.roiHeight;
                data->dstDimensions[index + f].width = data->dstBatch_width[n];
                data->dstDimensions[index + f].height = data->dstBatch_width[n];
            }
        }
    }
    if (data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU) {
        // STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8) {
            std::cerr << "UINT8888888888\n";
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
            std::cerr << "FLOAT32222222\n";

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8 && data->out_tensor_type == vx_type_e::VX_TYPE_INT8) {
            std::cerr << "INT888888888888888\n";

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
            std::cerr << "UINt8 TO FLOAT32\n";

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if(in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        // }
    }
    return status;
}

static vx_status VX_CALLBACK validateRotate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #10 type=%d (must be size)\n", scalar_type);
    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[NUM_OF_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 2);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processRotate(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    RotateLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        refreshRotate(node, parameters, num, data);
        // if (df_image == VX_DF_IMAGE_U8)
        // {
        //     rpp_status = rppi_rotate_u8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->kernelSize, data->nbatchSize, data->rppHandle);
        // }
        // else if (df_image == VX_DF_IMAGE_RGB)
        // {


            // rpp_status = rppi_rotate_u8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->kernelSize, data->nbatchSize, data->rppHandle);
        // }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        refreshRotate(node, parameters, num, data);
        // if (df_image == VX_DF_IMAGE_U8)
        // {
        //     rpp_status = rppi_rotate_u8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->kernelSize, data->nbatchSize, data->rppHandle);
        // }
        // else if (df_image == VX_DF_IMAGE_RGB)
        // {


            // rpp_status = rppi_rotate_u8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->kernelSize, data->nbatchSize, data->rppHandle);
        // }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshRotate(node, parameters, num, data);
        for(int i=0;i<data->nbatchSize;i++)
        {
            data->kernelSize[i]=90;
        }
        std::cerr<<"angle<<<<<<<<<<<<<<<<<<<<<<<<<<  "<<data->kernelSize[0];
        rpp_status = rppi_rotate_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst,data->srcDimensions, data->maxSrcDimensions, data->kernelSize,data->OutputFormatToggle, data->nbatchSize, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeRotate(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RotateLocalData *data = new RotateLocalData;
    unsigned roiType;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[10], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[9], &data->nbatchSize));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &roiType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->OutputFormatToggle, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    if (roiType == 1)
        data->roiType = RpptRoiType::XYWH;
    else
        data->roiType = RpptRoiType::LTRB;
    std::cerr << "\n layout " << data->layout;
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8) {
        std::cerr << "UUUUUUNIT8\n";
        data->src_desc_ptr->dataType = RpptDataType::U8;
    } else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
        std::cerr << "FFFFFFFLOAT\n";
        data->src_desc_ptr->dataType = RpptDataType::F32;
    }
    // else if (data->in_tensor_type->dataType == vx_type_e::VX_TYPE_FLOAT16)
    // {
    //     data->src_desc_ptr->dataType = RpptDataType::F16;
    // }
    else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8) {
        std::cerr << "iiiiin\n";
        data->src_desc_ptr->dataType = RpptDataType::I8;
    }
    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
        std::cerr << "ooooooooooooooFFFFFFFLOST\n";

        data->dst_desc_ptr->dataType = RpptDataType::F32;

    } else if (data->out_tensor_type == vx_type_e::VX_TYPE_UINT8) {
        std::cerr << "ooooooooooooooooooUUUUUUNIT8\n";

        data->dst_desc_ptr->dataType = RpptDataType::U8;
    } else if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
        std::cerr << "ooooooooooooooFFFFFFFLOST\n";

        data->dst_desc_ptr->dataType = RpptDataType::F32;

    }
    // else if (data->src_desc_ptr->dataType == vx_type_e::VX_TYPE_FLOAT16)
    // {
    //     data->src_desc_ptr->dataType = RpptDataType::F16;
    // }
    else if (data->out_tensor_type == vx_type_e::VX_TYPE_INT8) {
        std::cerr << "dst datatype check INT8";

        data->dst_desc_ptr->dataType = RpptDataType::I8;
    }
    data->src_desc_ptr->offsetInBytes = 0;
    if(data->layout == 0)
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->maxSrcDimensions.height = data->in_tensor_dims[1];
        data->maxSrcDimensions.width = data->in_tensor_dims[2];

        data->maxDstDimensions.height = data->out_tensor_dims[1];
        data->maxDstDimensions.width = data->out_tensor_dims[2];
    }
    else if(data->layout == 1)
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->maxSrcDimensions.height = data->in_tensor_dims[2];
        data->maxSrcDimensions.width = data->in_tensor_dims[3];

        data->maxDstDimensions.height = data->out_tensor_dims[1];
        data->maxDstDimensions.width = data->out_tensor_dims[2];
    }
    else if(data->layout == 2)
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
        data->maxSrcDimensions.height = data->in_tensor_dims[2];
        data->maxSrcDimensions.width = data->in_tensor_dims[3];

        data->maxDstDimensions.height = data->out_tensor_dims[2];
        data->maxDstDimensions.width = data->out_tensor_dims[3];
    }
    else if(data->layout == 3)
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
        data->maxSrcDimensions.height = data->in_tensor_dims[3];
        data->maxSrcDimensions.width = data->in_tensor_dims[4];

        data->maxDstDimensions.height = data->out_tensor_dims[3];
        data->maxDstDimensions.width = data->out_tensor_dims[4];
    }
    // if (data->layout == 0)  // NHWC
    // {
        // data->src_desc_ptr->n = data->in_tensor_dims[0];
        // data->src_desc_ptr->h = data->in_tensor_dims[1];
        // data->src_desc_ptr->w = data->in_tensor_dims[2];
        // data->src_desc_ptr->c = data->in_tensor_dims[3];
        // std::cerr << "\n n h w c " << data->src_desc_ptr->n << " " << data->src_desc_ptr->h << " " << data->src_desc_ptr->w << " " << data->src_desc_ptr->c;
        // data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        // data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
        // data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
        // data->src_desc_ptr->strides.cStride = 1;
        // data->src_desc_ptr->layout = RpptLayout::NHWC;
        // std::cerr << "\n Setting layoutt " << data->src_desc_ptr->layout;
        // std::cerr << "\n Setting data type " << data->src_desc_ptr->dataType;

    //     // destination_description_ptr
    //     data->dst_desc_ptr->n = data->out_tensor_dims[0];
    //     data->dst_desc_ptr->h = data->out_tensor_dims[1];
    //     data->dst_desc_ptr->w = data->out_tensor_dims[2];
    //     data->dst_desc_ptr->c = data->out_tensor_dims[3];
    //     std::cerr << "\n dest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    //     data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    //     data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.cStride = 1;
    //     data->dst_desc_ptr->layout = RpptLayout::NHWC;
    // } else if (data->layout == 1)  // NCHW
    // {
    //     data->src_desc_ptr->n = data->in_tensor_dims[0];
    //     data->src_desc_ptr->h = data->in_tensor_dims[2];
    //     data->src_desc_ptr->w = data->in_tensor_dims[3];
    //     data->src_desc_ptr->c = data->in_tensor_dims[1];
    //     data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    //     data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
    //     data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
    //     data->src_desc_ptr->strides.wStride = 1;
    //     data->src_desc_ptr->layout = RpptLayout::NCHW;

    //     data->dst_desc_ptr->n = data->out_tensor_dims[0];
    //     data->dst_desc_ptr->h = data->out_tensor_dims[2];
    //     data->dst_desc_ptr->w = data->out_tensor_dims[3];
    //     data->dst_desc_ptr->c = data->out_tensor_dims[1];
    //     std::cerr << "\ndest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    //     data->dst_desc_ptr->strides.cStride = data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    //     data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->w;
    //     data->dst_desc_ptr->strides.wStride = 1;
    //     data->dst_desc_ptr->layout = RpptLayout::NCHW;
    // } else if (data->layout == 2)  // NFHWC
    // {
    //     data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
    //     data->src_desc_ptr->h = data->in_tensor_dims[2];
    //     data->src_desc_ptr->w = data->in_tensor_dims[3];
    //     data->src_desc_ptr->c = data->in_tensor_dims[4];
    //     std::cerr << "\n n h w c " << data->src_desc_ptr->n << " " << data->src_desc_ptr->h << " " << data->src_desc_ptr->w << " " << data->src_desc_ptr->c;
    //     data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    //     data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
    //     data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
    //     data->src_desc_ptr->strides.cStride = 1;
    //     data->src_desc_ptr->layout = RpptLayout::NHWC;
    //     std::cerr << "\n Setting layouttttttttttttttt " << data->src_desc_ptr->layout;
    //     std::cerr << "\n Setting data type " << data->src_desc_ptr->dataType;

    //     // destination_description_ptr
    //     data->dst_desc_ptr->n = data->out_tensor_dims[0] * data->in_tensor_dims[1];
    //     data->dst_desc_ptr->h = data->out_tensor_dims[1];
    //     data->dst_desc_ptr->w = data->out_tensor_dims[2];
    //     data->dst_desc_ptr->c = data->out_tensor_dims[3];
    //     std::cerr << "\n dest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    //     data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    //     data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.cStride = 1;
    //     data->dst_desc_ptr->layout = RpptLayout::NHWC;
    // } else if (data->layout == 3)  // NFCHW
    // {
    //     data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
    //     data->src_desc_ptr->h = data->in_tensor_dims[3];
    //     data->src_desc_ptr->w = data->in_tensor_dims[4];
    //     data->src_desc_ptr->c = data->in_tensor_dims[2];
    //     data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    //     data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
    //     data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
    //     data->src_desc_ptr->strides.wStride = 1;
    //     data->src_desc_ptr->layout = RpptLayout::NCHW;

    //     data->dst_desc_ptr->n = data->out_tensor_dims[0] * data->in_tensor_dims[1];
    //     data->dst_desc_ptr->h = data->out_tensor_dims[1];
    //     data->dst_desc_ptr->w = data->out_tensor_dims[2];
    //     data->dst_desc_ptr->c = data->out_tensor_dims[3];
    //     std::cerr << "\n dest n h w c " << data->dst_desc_ptr->n << " " << data->dst_desc_ptr->h << " " << data->dst_desc_ptr->w << " " << data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    //     data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    //     data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    //     data->dst_desc_ptr->strides.cStride = 1;
    //     data->dst_desc_ptr->layout = RpptLayout::NCHW;
    // }
    // data->OutputFormatToggle=0;
    data->roi_tensor_Ptr = (RpptROI *)calloc(data->src_desc_ptr->n, sizeof(RpptROI));
    data->kernelSize = (vx_float32 *)malloc(sizeof(vx_float32) * data->src_desc_ptr->n);
    data->dstBatch_width = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);
    data->dstBatch_height = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);

    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->src_desc_ptr->n);
    data->dstDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->src_desc_ptr->n);
    data->srcBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->src_desc_ptr->n);
    data->srcBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->src_desc_ptr->n);
    refreshRotate(node, parameters, num, data);
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

static vx_status VX_CALLBACK uninitializeRotate(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RotateLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->roi_tensor_Ptr);
    free(data->srcDimensions);
    free(data->dstDimensions);

    free(data->srcBatch_width);
    free(data->srcBatch_height);
    free(data->kernelSize);
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,               // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity  // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
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

vx_status Rotate_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Rotate",
                                       VX_KERNEL_RPP_ROTATE,
                                       processRotate,
                                       11,
                                       validateRotate,
                                       initializeRotate,
                                       uninitializeRotate);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}