/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#define NUM_OF_DIMS 4

struct CropMirrorNormalizeLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    RppiSize *srcDimensions;
    RppiSize maxSrcDimensions;
    RppiSize *dstDimensions;
    RppiSize maxDstDimensions;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *start_x;
    vx_uint32 *start_y;
    vx_float32 *mean;
    vx_float32 *std_dev;
    vx_uint32 *mirror;
    vx_bool is_packed;                  // if true NHWC else NCHW
    size_t in_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    size_t out_tensor_dims[NUM_OF_DIMS];
    vx_uint32 channels;
    vx_uint32 batch_size;
    vx_uint32 chnShift; //NHWC to NCHW
    Rpp32u *srcBatch_width;
    Rpp32u *srcBatch_height;
    Rpp32u *dstBatch_width;
    Rpp32u *dstBatch_height;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};
/*
* Number of Dims is 4
* If is_packed is true - NHWC
* Dims[0] = N , Dims[1] = H, Dims[2] = W, Dims[3] = C
* If is_packed is true - NCHW
* Dims[0] = N , Dims[1] = C, Dims[2] = H, Dims[3] = W
*/
static vx_status VX_CALLBACK refreshCropMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num, CropMirrorNormalizeLocalData *data)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->batch_size, sizeof(vx_uint32), data->start_x, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[7], 0, data->batch_size, sizeof(vx_uint32), data->start_y, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[8], 0, data->batch_size, sizeof(vx_float32), data->mean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[9], 0, data->batch_size, sizeof(vx_float32), data->std_dev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[10], 0, data->batch_size, sizeof(vx_uint32), data->mirror, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &data->is_packed));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[12], &data->chnShift));
    if (data->is_packed)
    {
        data->maxSrcDimensions.height = data->in_tensor_dims[1];
        data->maxSrcDimensions.width = data->in_tensor_dims[2];
        data->maxDstDimensions.height = data->out_tensor_dims[1];
        data->maxDstDimensions.width = data->out_tensor_dims[2];
        data->channels = data->in_tensor_dims[3];
    }
    else
    {
        data->maxSrcDimensions.height = data->in_tensor_dims[2];
        data->maxSrcDimensions.width = data->in_tensor_dims[3];
        data->maxDstDimensions.height = data->out_tensor_dims[2];
        data->maxDstDimensions.width = data->out_tensor_dims[3];
        data->channels = data->in_tensor_dims[1];
    }
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->batch_size, sizeof(Rpp32u), data->srcBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->batch_size, sizeof(Rpp32u), data->srcBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->batch_size, sizeof(Rpp32u), data->dstBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->batch_size, sizeof(Rpp32u), data->dstBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->batch_size; i++)
    {
        data->srcDimensions[i].width = data->srcBatch_width[i];
        data->srcDimensions[i].height = data->srcBatch_height[i];
        data->dstDimensions[i].width = data->dstBatch_width[i];
        data->dstDimensions[i].height = data->dstBatch_height[i];
    }
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {

        vx_enum in_tensor_type = vx_type_e::VX_TYPE_UINT8;
        vx_enum out_tensor_type = vx_type_e::VX_TYPE_UINT8;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_tensor_type, sizeof(in_tensor_type)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &out_tensor_type, sizeof(out_tensor_type)));
        if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float16)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        // }
        else if (in_tensor_type == vx_type_e::VX_TYPE_INT8 && out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
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

static vx_status VX_CALLBACK validateCropMirrorNormalize(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_BOOL)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #11 type=%d (must be a boolean size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[12], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #12 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[13], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #13 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[14], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #14 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[4];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 3);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processCropMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    // std::cerr<<"\n processCropMirrorNormalize";
    vx_status vxstatus;
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    CropMirrorNormalizeLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_enum in_tensor_type = vx_type_e::VX_TYPE_UINT8;
    vx_enum out_tensor_type = vx_type_e::VX_TYPE_UINT8;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_tensor_type, sizeof(in_tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &out_tensor_type, sizeof(out_tensor_type)));
    Rpp32u N, C;
    N = data->batch_size;
    C = data->channels;

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        vxstatus = refreshCropMirrorNormalize(node, parameters, num, data);
        if (vxstatus != VX_SUCCESS)
        {
            return vxstatus;
        }
        if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                        data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                        data->start_x, data->start_y, data->mean,
                                                                        data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                         data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                         data->start_x, data->start_y, data->mean,
                                                                         data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_f32_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    int max_src_height, max_src_width;
                    max_src_width = data->maxSrcDimensions.width;
                    max_src_height = data->maxSrcDimensions.height;
                    // std::cerr<<"\n Gonna call rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_GPU";
                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                                data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                data->start_x, data->start_y, data->mean,
                                                                                data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {

                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                                data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                data->start_x, data->start_y, data->mean,
                                                                                data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_INT8 && out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                        data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                        data->start_x, data->start_y, data->mean,
                                                                        data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_i8_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                  data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                  data->start_x, data->start_y, data->mean,
        //                                                                  data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {
        //         if (data->is_packed)
        //         {
        //             rpp_status = rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                      data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                      data->start_x, data->start_y, data->mean,
        //                                                                      data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //         else
        //         {
        //             rpp_status = rppi_crop_mirror_normalize_f16_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                      data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                      data->start_x, data->start_y, data->mean,
        //                                                                      data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //     }
        // }
        // else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                   data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                   data->start_x, data->start_y, data->mean,
        //                                                                   data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {
        //         if (data->is_packed)
        //         {
        //             int max_src_height, max_src_width;
        //             max_src_width = data->maxSrcDimensions.width;
        //             max_src_height = data->maxSrcDimensions.height;
        //             // std::cerr<<"\n Gonna call rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_GPU";
        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions,(void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //         else
        //         {

        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, (void *)data->cl_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //     }
        // }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        vxstatus = refreshCropMirrorNormalize(node, parameters, num, data);
        if (vxstatus != VX_SUCCESS)
        {
            return vxstatus;
        }
        if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                        data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                        data->start_x, data->start_y, data->mean,
                                                                        data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                         data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                         data->start_x, data->start_y, data->mean,
                                                                         data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_f32_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    int max_src_height, max_src_width;
                    max_src_width = data->maxSrcDimensions.width;
                    max_src_height = data->maxSrcDimensions.height;
                    // std::cerr<<"\n Gonna call rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_GPU";
                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                                data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                data->start_x, data->start_y, data->mean,
                                                                                data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {

                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                                data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                data->start_x, data->start_y, data->mean,
                                                                                data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_INT8 && out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                        data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                        data->start_x, data->start_y, data->mean,
                                                                        data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                {
                    rpp_status = rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {
                    rpp_status = rppi_crop_mirror_normalize_i8_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
                                                                            data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
                                                                            data->start_x, data->start_y, data->mean,
                                                                            data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                  data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                  data->start_x, data->start_y, data->mean,
        //                                                                  data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {
        //         if (data->is_packed)
        //         {
        //             rpp_status = rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                      data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                      data->start_x, data->start_y, data->mean,
        //                                                                      data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //         else
        //         {
        //             rpp_status = rppi_crop_mirror_normalize_f16_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                      data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                      data->start_x, data->start_y, data->mean,
        //                                                                      data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //     }
        // }
        // else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                   data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                   data->start_x, data->start_y, data->mean,
        //                                                                   data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {
        //         if (data->is_packed)
        //         {
        //             int max_src_height, max_src_width;
        //             max_src_width = data->maxSrcDimensions.width;
        //             max_src_height = data->maxSrcDimensions.height;
        //             // std::cerr<<"\n Gonna call rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_GPU";
        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions,(void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //         else
        //         {

        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, (void *)data->hip_pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //     }
        // }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        vxstatus = refreshCropMirrorNormalize(node, parameters, num, data);
        if (vxstatus != VX_SUCCESS)
            return vxstatus;
        // std::cerr<<"\n CMN Tensor in host";
        if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_pln1_batchPD_host(data->pSrc, data->srcDimensions,
                                                                         data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                         data->start_x, data->start_y, data->mean,
                                                                         data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                    rpp_status = rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                else
                    rpp_status = rppi_crop_mirror_normalize_u8_pln3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_INT8 && out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_i8_pln1_batchPD_host(data->pSrc, data->srcDimensions,
                                                                         data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                         data->start_x, data->start_y, data->mean,
                                                                         data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                    rpp_status = rppi_crop_mirror_normalize_i8_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                else
                    rpp_status = rppi_crop_mirror_normalize_i8_pln3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_f32_pln1_batchPD_host(data->pSrc, data->srcDimensions,
                                                                          data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                          data->start_x, data->start_y, data->mean,
                                                                          data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {

                // std::cerr<<"\n CMN Tensor in IP: u8 OP: FP32";
                if (data->is_packed)
                    rpp_status = rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                              data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                              data->start_x, data->start_y, data->mean,
                                                                              data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                else
                    rpp_status = rppi_crop_mirror_normalize_f32_pln3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                              data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                              data->start_x, data->start_y, data->mean,
                                                                              data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {

                // std::cerr<<"\n CMN Tensor in IP: u8 OP: FP32";
                if (data->is_packed)
                {

                    std::cerr << "\n Gonna call rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_host";
                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                                 data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                 data->start_x, data->start_y, data->mean,
                                                                                 data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
                else
                {

                    rpp_status = rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                                 data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                                 data->start_x, data->start_y, data->mean,
                                                                                 data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                }
            }
        }
        else if (in_tensor_type == vx_type_e::VX_TYPE_INT8 && out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {
            if (C == 1)
            {
                rpp_status = rppi_crop_mirror_normalize_i8_pln1_batchPD_host(data->pSrc, data->srcDimensions,
                                                                         data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                         data->start_x, data->start_y, data->mean,
                                                                         data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
            else
            {
                if (data->is_packed)
                    rpp_status = rppi_crop_mirror_normalize_i8_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
                else
                    rpp_status = rppi_crop_mirror_normalize_i8_pln3_batchPD_host(data->pSrc, data->srcDimensions,
                                                                             data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
                                                                             data->start_x, data->start_y, data->mean,
                                                                             data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
            }
        }
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_f16_pln1_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                   data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                   data->start_x, data->start_y, data->mean,
        //                                                                   data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {
        //         // std::cerr<<"\n CMN Tensor in IP: 16 OP: 16";
        //         if (data->is_packed)
        //             rpp_status = rppi_crop_mirror_normalize_f16_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         else
        //             rpp_status = rppi_crop_mirror_normalize_f16_pln3_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        // }
        // else if (in_tensor_type == vx_type_e::VX_TYPE_UINT8 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     if (C == 1)
        //     {
        //         rpp_status = rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                   data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                   data->start_x, data->start_y, data->mean,
        //                                                                   data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //     }
        //     else
        //     {

        //         // std::cerr<<"\n CMN Tensor in IP: u8 OP: FP16";
        //         if (data->is_packed)
        //         {

        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //         else
        //         {

        //             rpp_status = rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_host(data->pSrc, data->srcDimensions,
        //                                                                       data->maxSrcDimensions, data->pDst, data->dstDimensions, data->maxDstDimensions,
        //                                                                       data->start_x, data->start_y, data->mean,
        //                                                                       data->std_dev, data->mirror, data->chnShift, N, data->rppHandle);
        //         }
        //     }
        // }

        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeCropMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status;
    CropMirrorNormalizeLocalData *data = new CropMirrorNormalizeLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[14], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[13], &data->batch_size));
    vx_size num_of_dims = NUM_OF_DIMS;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, data->in_tensor_dims, sizeof(vx_size) * num_of_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, data->out_tensor_dims, sizeof(vx_size) * num_of_dims));
    data->start_x = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->batch_size);
    data->start_y = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->batch_size);
    data->mean = (vx_float32 *)malloc(sizeof(vx_float32) * data->batch_size);
    data->std_dev = (vx_float32 *)malloc(sizeof(vx_float32) * data->batch_size);
    data->mirror = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->batch_size);
    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->batch_size);
    data->dstDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->batch_size);
    data->srcBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->batch_size);
    data->srcBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->batch_size);
    data->dstBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->batch_size);
    data->dstBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->batch_size);
    status = refreshCropMirrorNormalize(node, parameters, num, data);
    if (status != VX_SUCCESS)
    {
        std::cerr << "\n refreshCropMirrorNormalize exited with status :: " << status;

        return status;
    }
#if ENABLE_OPENCL
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.cmdq, data->batch_size);
    std::cerr << "\n Finished rppCreateWithStreamAndBatchSize";
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->batch_size);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeCropMirrorNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    CropMirrorNormalizeLocalData *data;
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

vx_status CropMirrorNormalize_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.CropMirrorNormalize",
                                       VX_KERNEL_RPP_CROPMIRRORNORMALIZE,
                                       processCropMirrorNormalize,
                                       15,
                                       validateCropMirrorNormalize,
                                       initializeCropMirrorNormalize,
                                       uninitializeCropMirrorNormalize);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 14, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
