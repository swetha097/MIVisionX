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

#ifndef MIVISIONX_RALI_API_AUGMENTATION_H
#define MIVISIONX_RALI_API_AUGMENTATION_H
#include "rali_api_types.h"

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
extern "C" RaliTensor RALI_API_CALL raliBrightness(RaliContext context, RaliTensor input, bool is_output,
                                                   RaliFloatParam alpha = NULL, RaliFloatParam beta = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C" RaliTensor RALI_API_CALL raliBrightnessFixed(RaliContext context, RaliTensor input,
                                                        float alpha, float beta,
                                                        bool is_output);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
extern "C" RaliTensor RALI_API_CALL raliGamma(RaliContext context, RaliTensor input,
                                              bool is_output,
                                              RaliFloatParam alpha = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C" RaliTensor RALI_API_CALL raliGammaFixed(RaliContext context, RaliTensor input, float alpha, bool is_output);

extern "C" RaliTensor RALI_API_CALL raliCopyTensor(RaliContext context, RaliTensor input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C" RaliTensor RALI_API_CALL raliNopTensor(RaliContext context, RaliTensor input, bool is_output);


extern "C" RaliTensor RALI_API_CALL raliCropMirrorNormalizeTensor(RaliContext context, RaliTensor input,
                                                                  RaliTensorLayout rali_tensor_layout,
                                                                  RaliTensorOutputType rali_tensor_output_type,
                                                                  unsigned crop_depth,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  float start_z,
                                                                  std::vector<float> &mean,
                                                                  std::vector<float> &std_dev,
                                                                  bool is_output,
                                                                  RaliIntParam mirror = NULL);


#endif //MIVISIONX_RALI_API_AUGMENTATION_H