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

#ifndef _VX_KERNELS_RPP_H_
#define _VX_KERNELS_RPP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#define VX_LIBRARY_RPP 5

    enum vx_kernel_ext_amd_rpp_e
    {
        VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x0,
        VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1,
        VX_KERNEL_RPP_ACCUMULATEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2,
        VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3,
        VX_KERNEL_RPP_ADDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4,
        VX_KERNEL_RPP_BLENDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x5,
        VX_KERNEL_RPP_BLURBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x6,
        VX_KERNEL_RPP_BITWISEANDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x7,
        VX_KERNEL_RPP_BITWISENOTBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x8,
        VX_KERNEL_RPP_BRIGHTNESSBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xa,
        VX_KERNEL_RPP_BOXFILTERBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xb,
        VX_KERNEL_RPP_CONTRASTBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xc,
        VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xd,
        VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xe,
        VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0xf,
        VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x10,
        VX_KERNEL_RPP_CANNYEDGEDETECTOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x11,
        VX_KERNEL_RPP_COLORTWISTBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x13,
        VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x14,
        VX_KERNEL_RPP_CROPPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x15,
        VX_KERNEL_RPP_COPY = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x16,
        VX_KERNEL_RPP_DILATEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x17,
        VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x18,
        VX_KERNEL_RPP_EXPOSUREBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x19,
        VX_KERNEL_RPP_EXCLUSIVEORBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1a,
        VX_KERNEL_RPP_ERODEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1b,
        VX_KERNEL_RPP_FLIPBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1c,
        VX_KERNEL_RPP_FOGBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1d,
        VX_KERNEL_RPP_FISHEYEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1e,
        VX_KERNEL_RPP_FASTCORNERDETECTOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x1f,
        VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x20,
        VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x21,
        VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x22,
        VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x23,
        VX_KERNEL_RPP_HISTOGRAM = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x24,
        VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x25,
        VX_KERNEL_RPP_HUEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x26,
        VX_KERNEL_RPP_HARRISCORNERDETECTOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x27,
        VX_KERNEL_RPP_INCLUSIVEORBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x28,
        VX_KERNEL_RPP_JITTERBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x29,
        VX_KERNEL_RPP_LENSCORRECTIONBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2a,
        VX_KERNEL_RPP_LOOKUPTABLEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2b,
        VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2c,
        VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2d,
        VX_KERNEL_RPP_MAGNITUDEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2e,
        VX_KERNEL_RPP_MULTIPLYBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x2f,
        VX_KERNEL_RPP_MAXBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x30,
        VX_KERNEL_RPP_MINBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x31,
        VX_KERNEL_RPP_MINMAXLOC = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x32,
        VX_KERNEL_RPP_MEANSTDDEV = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x33,
        VX_KERNEL_RPP_MEDIANFILTERBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x34,
        VX_KERNEL_RPP_NOISEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x35,
        VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x36,
        VX_KERNEL_RPP_NONLINEARFILTERBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x37,
        VX_KERNEL_RPP_NOP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x38,
        VX_KERNEL_RPP_PIXELATEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x39,
        VX_KERNEL_RPP_PHASEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3a,
        VX_KERNEL_RPP_RANDOMSHADOWBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3b,
        VX_KERNEL_RPP_RAINBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3c,
        VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3d,
        VX_KERNEL_RPP_RESIZEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3e,
        VX_KERNEL_RPP_RESIZECROPBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x3f,
        VX_KERNEL_RPP_ROTATEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x40,
        VX_KERNEL_RPP_REMAP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x41,
        VX_KERNEL_RPP_RESIZECROPMIRRORPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x42,
        VX_KERNEL_RPP_SNOWBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x43,
        VX_KERNEL_RPP_SUBTRACTBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x44,
        VX_KERNEL_RPP_SCALEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x45,
        VX_KERNEL_RPP_SATURATIONBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x46,
        VX_KERNEL_RPP_SOBELBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x47,
        VX_KERNEL_RPP_THRESHOLDINGBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x48,
        VX_KERNEL_RPP_TENSORADD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x49,
        VX_KERNEL_RPP_TENSORSUBTRACT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4a,
        VX_KERNEL_RPP_TENSORMULTIPLY = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4b,
        VX_KERNEL_RPP_TENSORMATRIXMULTIPLY = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4c,
        VX_KERNEL_RPP_TENSORLOOKUP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4d,
        VX_KERNEL_RPP_VIGNETTEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4e,
        VX_KERNEL_RPP_WARPAFFINEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x4f,
        VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x50,
    };

#ifdef __cplusplus
}
#endif

#endif
