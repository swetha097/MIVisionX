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

/*! \brief Converts Rali Memory type to OpenVX memory type
 *
 * @param mem input Rali type
 * @return the OpenVX type associated with input argument
 */
extern vx_enum vx_mem_type(RaliMemType mem);


/*! \brief Holds the information about an OpenVX Tensor */

struct TensorInfo
{
    friend struct Tensor;
    enum class Type
    {
        UNKNOWN = -1,
        REGULAR =0,
        VIRTUAL = 1,
        HANDLE =2
    };
    //! Default constructor,
    /*! initializes memory type to host  */
    TensorInfo();

    //! Initializer constructor
    TensorInfo(
        unsigned width_,
        unsigned height_,
        unsigned batches_,
        unsigned channels_,
        RaliMemType mem_type_,
        RaliColorFormat color_format,
        RaliTensorDataType data_type,
        RaliTensorFormat   tensor_format);

    unsigned width() const { return _width; }
    unsigned height_batch() const {return _height * _batch_size; }
    unsigned height() const { return _height; }
    unsigned channels() const { return _channels; }
    unsigned stride() const { return _stride; }
    unsigned height_single() const { return _height;}
    void width(unsigned width) { _width = width; }
    void height(unsigned height) { _height = height; }
    void batch_size(unsigned batch_size) { _batch_size = batch_size; }
    void channels(unsigned channels) { _channels = channels; }
    void format(RaliTensorFormat format) { _format = format; }
    void color_format(RaliColorFormat color_fmt)  { _color_fmt = color_fmt; }
    void data_type(RaliTensorDataType data_type)  { _data_type = data_type; }
    Type type() const { return _type; }
    RaliTensorDataType data_type() const { return _data_type;}
    unsigned batch_size() const {return _batch_size;}
    RaliMemType mem_type() const { return _mem_type; }
    RaliTensorFormat format() const { return _format; }
    unsigned data_size() const { return _data_size; }
    RaliColorFormat color_format() const {return _color_fmt; }
    unsigned color_plane_count() const { return _color_planes; }
    unsigned get_roi_width(int batch_idx) const;
    unsigned get_roi_height(int batch_idx) const;
    uint32_t * get_roi_width() const;
    uint32_t * get_roi_height() const;
    const std::vector<uint32_t>& get_roi_width_vec() const;
    const std::vector<uint32_t>& get_roi_height_vec() const;
    bool is_image() const { return _is_image; }
    size_t data_type_size()
    {
        if(_data_type == RaliTensorDataType::FP32)
        {
            _data_type_size = sizeof(vx_float32);
        }
        else if(_data_type == RaliTensorDataType::FP16)
        {
            _data_type_size = sizeof(vx_int16); // have to change this to float 16
        }
        else if(_data_type == RaliTensorDataType::UINT8)
        {
            _data_type_size = sizeof(vx_uint8);
        }
        return _data_type_size;
    }
private:
    Type _type = Type::UNKNOWN;//!< image type, whether is virtual image, created from handle or is a regular image
    unsigned _width;//!< image width for a single image in the batch
    unsigned _height;//!< image height for a single image in the batch
    unsigned _batch_size;//!< the batch size (images in the batch are stacked on top of each other)
    unsigned _channels;//!< number of channels
    unsigned _data_size;//!< total size of the memory needed to keep the image's data in bytes including all planes
    RaliMemType _mem_type;//!< memory type, currently either OpenCL or Host
    RaliTensorDataType _data_type = RaliTensorDataType::FP32;
    RaliTensorFormat _format = RaliTensorFormat::NCHW;
    RaliColorFormat _color_fmt;//!< color format of the image
    unsigned _color_planes;//!< number of color planes
    unsigned _stride;//!< if different from width
    std::shared_ptr<std::vector<uint32_t>> _roi_width;//!< The actual image width stored in the buffer, it's always smaller than _width/_batch_size. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one and observed for all.
    std::shared_ptr<std::vector<uint32_t>> _roi_height;//!< The actual image height stored in the buffer, it's always smaller than _height. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one changes can be observed for all.
    void reallocate_tensor_roi_buffers();
    bool _is_image = false;
    size_t _data_type_size;
};

bool operator==(const TensorInfo& rhs, const TensorInfo& lhs);

/*! \brief Holds an OpenVX tensor and it's info
*
* Keeps the information about the RALI tensor that can be queried using OVX API as well,
* but for simplicity and ease of use, they are kept in separate fields
*/
struct Tensor
{
    int swap_handle(void* handle);

    const TensorInfo& info() { return _info; }
    //! Default constructor
    Tensor() = delete;
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
    ~Tensor();

    //! Constructor accepting the image information as input
    explicit Tensor(const TensorInfo& tensor_info);

    int create(vx_context context);
    void update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height);
    void reset_tensor_roi() { _info.reallocate_tensor_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);

private:
    vx_tensor _vx_handle = nullptr;//!< The OpenVX image
    void* _mem_handle = nullptr;//!< Pointer to the image's internal buffer (opencl or host)
    TensorInfo _info;//!< The structure holding the info related to the stored OpenVX image
    vx_context _context = nullptr;
};


