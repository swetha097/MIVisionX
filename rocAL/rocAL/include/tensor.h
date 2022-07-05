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

#pragma once

#include <VX/vx.h>
#include <VX/vx_types.h>
#include <vector>
#include <cstring>
#include <array>
#include <queue>
#include <memory>
#if ENABLE_HIP
#include "hip/hip_runtime.h"
#include "device_manager_hip.h"
#else
#include "device_manager.h"
#endif
#include "commons.h"
// to align tensor width
#define TENSOR_WIDTH_ALIGNMENT   0 // todo:: check to see if we need this

/*! \brief Converts Rocal Memory type to OpenVX memory type
 *
 * @param mem input Rocal type
 * @return the OpenVX type associated with input argument
 */
vx_enum vx_mem_type(RocalMemType mem); // TODO - extern was used previously ?

vx_size tensor_data_size(RocalTensorDataType data_type);

/*! \brief Holds the information about a rocALTensor */

class rocALTensorInfo
{
public:
    friend class rocALTensor;
    enum class Type
    {
        UNKNOWN = -1,
        REGULAR = 0,
        VIRTUAL = 1,
        HANDLE  = 2
    };
    // Default constructor
    rocALTensorInfo();

    //! Initializer constructor with only fields common to all types (Image/ Video / Audio)
    rocALTensorInfo(unsigned num_of_dims,
                    std::shared_ptr<std::vector<unsigned>> dims,
                    RocalMemType mem_type,
                    RocalTensorDataType data_type);

    // Setting properties required for Image / Video
    void set_roi_type(RocalROIType roi_type) { _roi_type = roi_type; }
    void set_data_type(RocalTensorDataType data_type)
    {
        int data_size = _data_size / _data_type_size;
        _data_type = data_type;
        _data_size = data_size * data_type_size();
    }
    int get_data_type()
    {
        if(_data_type== RocalTensorDataType::FP32)
        return 1;
        else 
        return 0;
    }
    void set_tensor_layout(RocalTensorlayout layout)
    {
        if(layout != RocalTensorlayout::NONE)
        {
            _is_image = true;
            if(layout == RocalTensorlayout::NHWC)
            {
                _max_width = _dims->at(2);
                _max_height = _dims->at(1);
            }
            else if(layout == RocalTensorlayout::NCHW)
            {
                _max_width = _dims->at(3);
                _max_height = _dims->at(2);
            }
            else if(layout == RocalTensorlayout::NFHWC)
            {
                _max_width = _dims->at(3);
                _max_height = _dims->at(2);
                _frames = _dims->at(1);
            }
            else if(layout == RocalTensorlayout::NFCHW)
            {
                _max_width = _dims->at(4);
                _max_height = _dims->at(3);
                _frames = _dims->at(1);
            }
            reallocate_tensor_roi_buffers();
        }
        else
        {
            _max_width = _dims->at(1);
            _max_height = _dims->at(2);
            // std::cerr<<"\n Setting _max_width :: "<<_max_width<<"\t _max_height :: "<<_max_height;
        }
        _layout = layout;
    }
    void set_color_format(RocalColorFormat color_format) { _color_format = color_format; }

    unsigned num_of_dims() const { return _num_of_dims; }
    unsigned batch_size() const { return _batch_size; }
    unsigned data_size() const { return _data_size; }
    unsigned max_width() const { return _max_width; }
    unsigned max_height() const { return _max_height; }

    void set_width(unsigned width) { _width = width; }
    void set_height(unsigned height) {_height= height; }
    unsigned get_width() const { return _width; }
    unsigned get_height() const { return _height; }


    std::shared_ptr<std::vector<unsigned>> dims() const { return _dims; }
    RocalMemType mem_type() const { return _mem_type; }
    RocalROIType roi_type() const { return _roi_type; }
    RocalTensorDataType data_type() const { return _data_type; }
    RocalTensorlayout layout() const { return _layout; }
    std::shared_ptr<std::vector<RocalROI>> get_roi() const { return _roi; }
    RocalColorFormat color_format() const {return _color_format; }
    Type type() const { return _type; }
    bool is_image() const { return _is_image; }
    unsigned data_type_size()
    {
        _data_type_size = tensor_data_size(_data_type);
        return _data_type_size;
    }
    unsigned num_of_frames() const
    {
        if(_num_of_dims == 5)
            return _frames;
        else
            ERR("The frames dimension 'F' is applicable only for 5D NFCHW or NFHWC tensors")
    }

private:
    Type _type = Type::UNKNOWN;//!< tensor type, whether is virtual tensor, created from handle or is a regular tensor
    unsigned _num_of_dims;
    std::shared_ptr<std::vector<unsigned>> _dims;
    unsigned _batch_size;
    RocalMemType _mem_type;
    RocalROIType _roi_type;
    RocalTensorDataType _data_type = RocalTensorDataType::FP32;
    RocalTensorlayout _layout = RocalTensorlayout::NCHW;
    RocalColorFormat _color_format;
    std::shared_ptr<std::vector<RocalROI>> _roi;
    unsigned _data_type_size = tensor_data_size(_data_type);
    unsigned _data_size = 0;
    unsigned _max_width, _max_height,_width,_height;
    unsigned _frames; // denotes the F dimension in the tensor
    bool _is_image = false;
    void reallocate_tensor_roi_buffers();
};

bool operator==(const rocALTensorInfo& rhs, const rocALTensorInfo& lhs);
class rocALTensor
{
public:
    int swap_handle(void* handle);

    const rocALTensorInfo& info() { return _info; }
    //! Default constructor
    rocALTensor() = delete;
    void* buffer() { return _mem_handle; }
    vx_tensor handle() { return _vx_handle; }
    vx_context context() { return _context; }
#if !ENABLE_HIP
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#else
    unsigned copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync);
    unsigned copy_data(hipStream_t stream, void* hip_memory, bool sync);
#endif
    //! Default destructor
    /*! Releases the OpenVX Tensor object */
    ~rocALTensor();

    //! Constructor accepting the tensor information as input
    explicit rocALTensor(const rocALTensorInfo& tensor_info);

    int create(vx_context context);
    void update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height);
    void reset_tensor_roi() { _info.reallocate_tensor_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);

private:
    vx_tensor _vx_handle = nullptr;//!< The OpenVX tensor
    void* _mem_handle = nullptr;//!< Pointer to the tensor's internal buffer (opencl or host)
    rocALTensorInfo _info;//!< The structure holding the info related to the stored OpenVX tensor
    vx_context _context = nullptr;
};

/*! \brief Contains a list of rocALTensors corresponding to different outputs */
class rocALTensorList
{
public:
    unsigned size() { return _tensor_list.size(); }
    bool empty() { return _tensor_list.empty(); }
    void push_back(rocALTensor * tensor)
    {
        _tensor_list.emplace_back(tensor);
        _tensor_data_size.emplace_back(tensor->info().data_size());
    }
    std::vector<size_t> data_size()
    {
        return _tensor_data_size;
    }
    void release()
    {
        for(auto& tensor: _tensor_list)
            delete tensor;
    }
    rocALTensor * operator[](size_t index)
    {
        return _tensor_list[index];
    }
    rocALTensor * at(size_t index)
    {
        return _tensor_list[index];
    }
    void operator=(rocALTensorList &other)
    {
        for(unsigned idx = 0; idx < other.size(); idx++)
        {
            auto* new_tensor = new rocALTensor(other[idx]->info());
            if (new_tensor->create_from_handle(other[idx]->context()) != 0)
                THROW("Cannot create the tensor from handle")
            this->push_back(new_tensor);
        }
    }

private:
    std::vector<rocALTensor*> _tensor_list;
    std::vector<size_t> _tensor_data_size;
};