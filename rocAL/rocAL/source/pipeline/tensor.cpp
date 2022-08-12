
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

#include <cstdio>
#if !ENABLE_HIP
#include <CL/cl.h>
#endif
#include <stdexcept>
#include <vx_ext_amd.h>
#include <cstring>
#include "commons.h"
#include "tensor.h"

vx_enum vx_mem_type(RocalMemType mem)
{
    switch(mem)
    {
        case RocalMemType::OCL:
        {
            return VX_MEMORY_TYPE_OPENCL;
        }
        break;
        case RocalMemType::HOST:
        {
            return VX_MEMORY_TYPE_HOST;
        }
        break;
        case RocalMemType::HIP:
        {
            return VX_MEMORY_TYPE_HIP;
        }
        break;
        default:
            throw std::runtime_error("Memory type not valid");
    }
}

vx_size tensor_data_size(RocalTensorDataType data_type)
{
    switch(data_type)
    {
        case RocalTensorDataType::FP32:
        {
            return sizeof(vx_float32);
        }
        break;
        case RocalTensorDataType::FP16:
        {
            return sizeof(vx_int16);
        }
        break;
        case RocalTensorDataType::UINT8:
        {
            return sizeof(vx_uint8);
        }
        case RocalTensorDataType::UINT32:
        {
            return sizeof(vx_uint32);
        }
        case RocalTensorDataType::INT32:
        {
            return sizeof(vx_int32);
        }
        break;
        default:
            throw std::runtime_error("tensor data_type not valid");
    }
}

bool operator==(const rocALTensorInfo &rhs, const rocALTensorInfo &lhs)
{
    return (rhs.dims() == lhs.dims() &&
            rhs.mem_type() == lhs.mem_type() &&
            rhs.data_type() == lhs.data_type() &&
            rhs.color_format() == lhs.color_format() &&
            rhs.layout() == lhs.layout());
}

void rocALTensorInfo::reallocate_tensor_roi_buffers()
{
    _roi = std::make_shared<std::vector<RocalROI>>(_batch_size);
    // _roi_height = std::make_shared<std::vector<uint32_t>>(_batch_size);
    // _roi_width = std::make_shared<std::vector<uint32_t>>(_batch_size);

    // TODO - Needs change here
    _roi->resize(_batch_size);
    if(layout() == RocalTensorlayout::NCHW)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi->at(i).x1 = 0;
            _roi->at(i).y1 = 0;
            _roi->at(i).x2 = _dims.at(3);
            _roi->at(i).y2 = _dims.at(2);
        }
    }
    else if(layout() == RocalTensorlayout::NHWC)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi->at(i).x1 = 0;
            _roi->at(i).y1 = 0;
            _roi->at(i).x2 = _dims.at(2);
            _roi->at(i).y2 = _dims.at(1);
        }
    }
    else if(layout() == RocalTensorlayout::NFCHW)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi->at(i).x1 = 0;
            _roi->at(i).y1 = 0;
            _roi->at(i).x2 = _dims.at(4);
            _roi->at(i).y2 = _dims.at(3);
        }
    }
    else if(layout() == RocalTensorlayout::NFHWC)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi->at(i).x1 = 0;
            _roi->at(i).y1 = 0;
            _roi->at(i).x2 = _dims.at(3);
            _roi->at(i).y2 = _dims.at(2);
        }
    }
}

rocALTensorInfo::rocALTensorInfo() : _type(Type::UNKNOWN),
                                _num_of_dims(0),
                                _dims({}),
                                _mem_type(RocalMemType::HOST),
                                _data_type(RocalTensorDataType::FP32){}

rocALTensorInfo::rocALTensorInfo(
    unsigned num_of_dims,
    std::vector<unsigned> dims,
    RocalMemType mem_type,
    RocalTensorDataType data_type) : _type(Type::UNKNOWN),
                                _num_of_dims(num_of_dims),
                                _dims(dims),
                                _mem_type(mem_type),
                                _data_type(data_type)
{
    vx_size data_size = tensor_data_size(data_type);
    // unsigned alignpixels = TENSOR_WIDTH_ALIGNMENT; // Check if needed
    _data_size = data_size;
    for(unsigned i = 0; i < _num_of_dims; i++)
    {
        _data_size *= dims.at(i);
    }
    if(_num_of_dims <= 3)
        _is_image = false;
    _batch_size = _dims.at(0);
    // if(_num_of_dims == 5)
    //     _batch_size *= _dims.at(1); // TODO - Fix for sequence reader need to check how we can check if it not video reader and update
    // initializing each Tensor dimension in the batch with the maximum Tensor size, they'll get updated later during the runtime
    // Update this only if the tensor is image
}

void rocALTensor::update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height)
{
    if(_info.is_image())
    {
        auto max_dims = _info.max_dims();
        unsigned max_width = max_dims.at(0);
        unsigned max_height = max_dims.at(1);

        if (width.size() != height.size())
            THROW("Batch size of Tensor height and width info does not match")

        if (width.size() != info().batch_size())
            THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))

        for (unsigned i = 0; i < info().batch_size(); i++)
        {
            if (width[i] > max_width)
            {
                ERR("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(max_width))
                _info.get_roi()->at(i).x2 = max_width;
            }
            else
            {
                _info.get_roi()->at(i).x2 = width[i];
            }
            if (height[i] > max_height)
            {
                ERR("Given ROI height is larger than buffer with for tensor[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(max_height))
                _info.get_roi()->at(i).y2 = max_height;
            }
            else
            {
                _info.get_roi()->at(i).y2 = height[i];
            }
        }
    }
}

rocALTensor::~rocALTensor()
{
    _mem_handle = nullptr;
    if(_vx_handle)
        vxReleaseTensor(&_vx_handle);
}

//! Converts the Rocal data_type to OpenVX
vx_enum interpret_tensor_data_type(RocalTensorDataType data_type)
{
    switch (data_type)
    {
        case RocalTensorDataType::FP32:
            return VX_TYPE_FLOAT32;
        case RocalTensorDataType::FP16:
            return VX_TYPE_FLOAT16;
        case RocalTensorDataType::UINT8:
            return VX_TYPE_UINT8;
        default:
            THROW("Unsupported Tensor type " + TOSTR(data_type))
    }
}

rocALTensor::rocALTensor(const rocALTensorInfo &tensor_info) : _info(tensor_info)
{
    _info._type = rocALTensorInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int rocALTensor::create_virtual(vx_context context, vx_graph graph)
{
    if (_vx_handle)
        return -1;

    _context = context;

    // TODO - find a better way to convert from unsigned to size_t
    unsigned num_of_dims = _info.num_of_dims();
    vx_size dims[num_of_dims];
    for(unsigned i = 0; i < num_of_dims; i++)
    {
        dims[i] = _info.dims().at(i);
    }
    _vx_handle = vxCreateVirtualTensor(graph, num_of_dims, dims, interpret_tensor_data_type(_info.data_type()), 0);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualTensor(input:[" + TOSTR(_info.max_dims().at(0)) + "W" + TOSTR(_info.max_dims().at(1)) + "H" + "]): failed " + TOSTR(status))

    _info._type = rocALTensorInfo::Type::VIRTUAL;
    return 0;
}

int rocALTensor::create_from_handle(vx_context context)
{
    if (_vx_handle)
    {
        WRN("Tensor object create method is already called ")
        return -1;
    }
    // if (_info.height() == 0 || _info.width() == 0 || _info.channels() == 0 || _info.batch_size() == 0)
    //     THROW("Invalid tensor dimension" + TOSTR(_info.width()) + " x " + TOSTR(_info.height()) + " x " + TOSTR(_info.channels()) + " x " + TOSTR(_info.batch_size()));

    _context = context;
    // bool nhwc = true;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    unsigned num_of_dims = _info.num_of_dims();
    vx_size stride[num_of_dims];
    void *ptr[1] = {nullptr};
    // bool nhwc = ((_info.layout() == RocalTensorlayout::NHWC) ? true: false); // TODO : Fiona

    stride[0] = tensor_data_size(_info.data_type());
    for(unsigned i = 1; i < num_of_dims; i++)
    {
        stride[i] = stride[i - 1] * _info.dims().at(i - 1);
    }
    vx_status status;
    // TODO - find a better way to convert from unsigned to size_t
    vx_size dims[num_of_dims];
    for(unsigned i = 0; i < num_of_dims; i++)
    {
        dims[i] = _info.dims().at(i);
    }
    _vx_handle = vxCreateTensorFromHandle(_context, num_of_dims, dims, tensor_data_type, 0, stride, ptr, vx_mem_type(_info._mem_type));
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(input: failed " + TOSTR(status))
    _info._type = rocALTensorInfo::Type::HANDLE;
    // shobi to be checked
    // if(_info._data_size == 0)
    //     _info._data_size = stride[3] * _info.channels(); // since data size is set while initializing info
    return 0;
}

int rocALTensor::create(vx_context context)
{
    if (_vx_handle)
        return -1;

    _context = context;

    vx_status status;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    _vx_handle = vxCreateTensor(context, _info.num_of_dims(),(vx_size*) _info.dims().data(), tensor_data_type, 0);
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor(input: failed " + TOSTR(status))
    _info._type = rocALTensorInfo::Type::REGULAR;
    return 0;
}
#if ENABLE_OPENCL
unsigned rocALTensor::copy_data(cl_command_queue queue, unsigned char *user_buffer, bool sync)
{
    if (_info._type != rocALTensorInfo::Type::HANDLE)
        return 0;

    if (_info._mem_type == RocalMemType::OCL)
    {

        cl_int status;
        if ((status = clEnqueueReadBuffer(queue,
                                          (cl_mem)_mem_handle,
                                          sync ? (CL_TRUE) : CL_FALSE,
                                          0,
                                          _info.data_size(),
                                          user_buffer,
                                          0, nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
    }
    else
    {
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return _info.data_size();
}
unsigned rocALTensor::copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync)
{
    return 0;
}
#elif ENABLE_HIP
    unsigned rocALTensor::copy_data(hipStream_t stream, void* host_memory, bool sync)
    {
        if(_info._type != rocALTensorInfo::Type::HANDLE)
            return 0;

        if (_info._mem_type == RocalMemType::HIP)
        {
            // copy from device to host
            hipError_t status;
            if ((status = hipMemcpyDtoHAsync((void *)host_memory, _mem_handle, _info.data_size(), stream)))
                THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
            if (sync)
            {
                if ((status =hipStreamSynchronize(stream)))
                    THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
            }
        }
        else
        {
            memcpy(host_memory, _mem_handle, _info.data_size());
        }
        return _info.data_size();
    }
#endif
unsigned rocALTensor::copy_data(void* user_buffer, bool sync)
{
    if(_info._type != rocALTensorInfo::Type::HANDLE)
        return 0;

    if (_info._mem_type == RocalMemType::HIP)
    {
#if ENABLE_HIP
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoH((void *)user_buffer, _mem_handle, _info.data_size())))
            THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
#endif
    }
    else
    {
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return _info.data_size();
}

int rocALTensor::swap_handle(void *handle)
{
    vx_status status;
    if ((status = vxSwapTensorHandle(_vx_handle, handle, nullptr)) != VX_SUCCESS)
    {
        ERR("Swap handles failed for tensor" + TOSTR(status));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}