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

#ifndef MIVISIONX_ROCAL_API_INFO_H
#define MIVISIONX_ROCAL_API_INFO_H
#include "rocal_api_types.h"

///
/// \param image
/// \return The width of the ROCAL's output image in pixels
extern "C" int ROCAL_API_CALL rocalGetOutputWidth(RocalContext rocal_context);

///
/// \param image
/// \return The height of the ROCAL's output image in pixels. It includes all images in the batch.
extern "C" int ROCAL_API_CALL rocalGetOutputHeight(RocalContext rocal_context);

///
/// \param rocal_context
/// \return The color format of the ROCAL's output. It's equivalent of what's passed to the loaders as input color format.
extern "C" int ROCAL_API_CALL rocalGetOutputColorFormat(RocalContext rocal_context);

///
/// \param rocal_context
/// \return The number of images yet to be processed
extern "C"  size_t  ROCAL_API_CALL rocalGetRemainingImages(RocalContext rocal_context);

/// Returned value valid only after rocalVerify is called
/// \param image
/// \return Width of the graph output image
extern "C" size_t ROCAL_API_CALL rocalGetImageWidth(RocalTensor image);

/// Returned value valid only after rocalVerify is called
/// \param image
/// \return Height of the pipeline output image, includes all images in the batch
extern "C" size_t ROCAL_API_CALL rocalGetImageHeight(RocalTensor image);


/// Returned value valid only after rocalVerify is called
/// \param image
/// \return Color format of the pipeline output image,
extern "C" size_t ROCAL_API_CALL rocalGetImagePlanes(RocalTensor image);

/// Returned value valid only after rocalVerify is called
/// \param rocal_context
/// \return 1 if all images have been processed, otherwise 0
extern "C" size_t ROCAL_API_CALL rocalIsEmpty(RocalContext rocal_context);


///
/// \param rocal_context
/// \return Number of augmentation graph branches. Defined by number of calls to the augmentation API's with the is_output flag set to true.
extern "C" size_t ROCAL_API_CALL rocalGetAugmentationBranchCount(RocalContext rocal_context);

///
/// \param rocal_context
/// \return The status of tha last API call
extern "C" RocalStatus ROCAL_API_CALL rocalGetStatus(RocalContext rocal_context);

///
/// \param rocal_context
/// \return The last error message generated by call to rocal API
extern "C" const char* ROCAL_API_CALL rocalGetErrorMessage(RocalContext rocal_context);

///
/// \param rocal_context
/// \return The timing info associated with recent execution.
extern "C" TimingInfo ROCAL_API_CALL rocalGetTimingInfo(RocalContext rocal_context);

#endif //MIVISIONX_ROCAL_API_INFO_H
