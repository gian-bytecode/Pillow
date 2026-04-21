/*
 * Pillow GPU Acceleration — OpenCL Backend
 *
 * Implements all GPU operations using OpenCL 1.2+.
 * Kernels are compiled at runtime from embedded source strings.
 *
 * Copyright (c) 2026 Pillow Contributors
 */

#ifdef HAVE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "GpuImaging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ================================================================== */
/* OpenCL kernel source strings                                        */
/* ================================================================== */

static const char *OPENCL_KERNEL_COMMON =
"/* Common utilities */\n"
"#define CLIP8(v) clamp((int)(v), 0, 255)\n"
"#define MULDIV255(a, b) (((a) * (b) + 128 + (((a) * (b) + 128) >> 8)) >> 8)\n"
"\n";

static const char *OPENCL_KERNEL_BLUR =
"/* Separable box blur — horizontal pass (per-pixel) */\n"
"__kernel void box_blur_h(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int pixelsize,\n"
"    int radius)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (x >= width || y >= height) return;\n"
"    int linesize = width * pixelsize;\n"
"    int diam = 2 * radius + 1;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        int acc = 0;\n"
"        for (int i = -radius; i <= radius; i++) {\n"
"            int sx = clamp(x + i, 0, width - 1);\n"
"            acc += input[y * linesize + sx * pixelsize + c];\n"
"        }\n"
"        output[y * linesize + x * pixelsize + c] =\n"
"            (uchar)((acc + diam/2) / diam);\n"
"    }\n"
"}\n"
"\n"
"/* Separable box blur — vertical pass (per-pixel) */\n"
"__kernel void box_blur_v(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int pixelsize,\n"
"    int radius)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (x >= width || y >= height) return;\n"
"    int linesize = width * pixelsize;\n"
"    int diam = 2 * radius + 1;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        int acc = 0;\n"
"        for (int i = -radius; i <= radius; i++) {\n"
"            int sy = clamp(y + i, 0, height - 1);\n"
"            acc += input[sy * linesize + x * pixelsize + c];\n"
"        }\n"
"        output[y * linesize + x * pixelsize + c] =\n"
"            (uchar)((acc + diam/2) / diam);\n"
"    }\n"
"}\n"
"\n"
"/* Gaussian blur separable pass — weighted */\n"
"__kernel void gaussian_blur_h(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const float *weights,\n"
"    int width, int height, int pixelsize,\n"
"    int radius)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (x >= width || y >= height) return;\n"
"    int linesize = width * pixelsize;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float sum = 0.0f;\n"
"        for (int i = -radius; i <= radius; i++) {\n"
"            int sx = clamp(x + i, 0, width - 1);\n"
"            sum += (float)input[y * linesize + sx * pixelsize + c]\n"
"                   * weights[i + radius];\n"
"        }\n"
"        output[y * linesize + x * pixelsize + c] =\n"
"            (uchar)clamp((int)(sum + 0.5f), 0, 255);\n"
"    }\n"
"}\n"
"\n"
"__kernel void gaussian_blur_v(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const float *weights,\n"
"    int width, int height, int pixelsize,\n"
"    int radius)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (x >= width || y >= height) return;\n"
"    int linesize = width * pixelsize;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float sum = 0.0f;\n"
"        for (int i = -radius; i <= radius; i++) {\n"
"            int sy = clamp(y + i, 0, height - 1);\n"
"            sum += (float)input[sy * linesize + x * pixelsize + c]\n"
"                   * weights[i + radius];\n"
"        }\n"
"        output[y * linesize + x * pixelsize + c] =\n"
"            (uchar)clamp((int)(sum + 0.5f), 0, 255);\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_FILTER =
"/* Generic NxN convolution kernel */\n"
"__kernel void convolve(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const float *kernel_data,\n"
"    int width, int height, int pixelsize,\n"
"    int kw, int kh,\n"
"    float divisor, float offset)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (x >= width || y >= height) return;\n"
"    int linesize = width * pixelsize;\n"
"    int kw2 = kw / 2;\n"
"    int kh2 = kh / 2;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float sum = 0.0f;\n"
"        for (int ky = 0; ky < kh; ky++) {\n"
"            int sy = clamp(y + ky - kh2, 0, height - 1);\n"
"            for (int kx = 0; kx < kw; kx++) {\n"
"                int sx = clamp(x + kx - kw2, 0, width - 1);\n"
"                float pixel = (float)input[sy * linesize + sx * pixelsize + c];\n"
"                sum += pixel * kernel_data[ky * kw + kx];\n"
"            }\n"
"        }\n"
"        int val = (int)(sum / divisor + offset + 0.5f);\n"
"        output[y * linesize + x * pixelsize + c] =\n"
"            (uchar)clamp(val, 0, 255);\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_RESAMPLE =
"/* Bilinear interpolation for resize */\n"
"__kernel void resample_bilinear(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float box_x0, float box_y0, float box_x1, float box_y1)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"\n"
"    float scale_x = (box_x1 - box_x0) / (float)out_w;\n"
"    float scale_y = (box_y1 - box_y0) / (float)out_h;\n"
"    float src_x = box_x0 + (ox + 0.5f) * scale_x - 0.5f;\n"
"    float src_y = box_y0 + (oy + 0.5f) * scale_y - 0.5f;\n"
"\n"
"    int x0 = (int)floor(src_x);\n"
"    int y0 = (int)floor(src_y);\n"
"    float fx = src_x - x0;\n"
"    float fy = src_y - y0;\n"
"    int x1 = x0 + 1;\n"
"    int y1 = y0 + 1;\n"
"    x0 = clamp(x0, 0, in_w - 1);\n"
"    x1 = clamp(x1, 0, in_w - 1);\n"
"    y0 = clamp(y0, 0, in_h - 1);\n"
"    y1 = clamp(y1, 0, in_h - 1);\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float v00 = input[y0 * in_linesize + x0 * pixelsize + c];\n"
"        float v10 = input[y0 * in_linesize + x1 * pixelsize + c];\n"
"        float v01 = input[y1 * in_linesize + x0 * pixelsize + c];\n"
"        float v11 = input[y1 * in_linesize + x1 * pixelsize + c];\n"
"        float top = v00 + (v10 - v00) * fx;\n"
"        float bot = v01 + (v11 - v01) * fx;\n"
"        float val = top + (bot - top) * fy;\n"
"        output[oy * out_linesize + ox * pixelsize + c] =\n"
"            (uchar)clamp((int)(val + 0.5f), 0, 255);\n"
"    }\n"
"}\n"
"\n"
"/* Bicubic interpolation */\n"
"float cubic_weight(float x) {\n"
"    float ax = fabs(x);\n"
"    if (ax < 1.0f) return (1.5f * ax - 2.5f) * ax * ax + 1.0f;\n"
"    if (ax < 2.0f) return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;\n"
"    return 0.0f;\n"
"}\n"
"\n"
"__kernel void resample_bicubic(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float box_x0, float box_y0, float box_x1, float box_y1)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"\n"
"    float scale_x = (box_x1 - box_x0) / (float)out_w;\n"
"    float scale_y = (box_y1 - box_y0) / (float)out_h;\n"
"    float src_x = box_x0 + (ox + 0.5f) * scale_x - 0.5f;\n"
"    float src_y = box_y0 + (oy + 0.5f) * scale_y - 0.5f;\n"
"\n"
"    int ix = (int)floor(src_x);\n"
"    int iy = (int)floor(src_y);\n"
"    float fx = src_x - ix;\n"
"    float fy = src_y - iy;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float sum = 0.0f;\n"
"        for (int j = -1; j <= 2; j++) {\n"
"            float wy = cubic_weight(fy - j);\n"
"            int sy = clamp(iy + j, 0, in_h - 1);\n"
"            for (int i = -1; i <= 2; i++) {\n"
"                float wx = cubic_weight(fx - i);\n"
"                int sx = clamp(ix + i, 0, in_w - 1);\n"
"                sum += input[sy * in_linesize + sx * pixelsize + c] * wx * wy;\n"
"            }\n"
"        }\n"
"        output[oy * out_linesize + ox * pixelsize + c] =\n"
"            (uchar)clamp((int)(sum + 0.5f), 0, 255);\n"
"    }\n"
"}\n"
"\n"
"/* Lanczos3 interpolation */\n"
"float sinc(float x) {\n"
"    if (fabs(x) < 1e-6f) return 1.0f;\n"
"    float px = M_PI_F * x;\n"
"    return sin(px) / px;\n"
"}\n"
"\n"
"float lanczos_weight(float x, int a) {\n"
"    float ax = fabs(x);\n"
"    if (ax >= (float)a) return 0.0f;\n"
"    return sinc(x) * sinc(x / (float)a);\n"
"}\n"
"\n"
"__kernel void resample_lanczos(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float box_x0, float box_y0, float box_x1, float box_y1)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"    int a = 3; /* Lanczos3 */\n"
"\n"
"    float scale_x = (box_x1 - box_x0) / (float)out_w;\n"
"    float scale_y = (box_y1 - box_y0) / (float)out_h;\n"
"    float support_x = fmax((float)a, (float)a * scale_x);\n"
"    float support_y = fmax((float)a, (float)a * scale_y);\n"
"    float src_x = box_x0 + (ox + 0.5f) * scale_x - 0.5f;\n"
"    float src_y = box_y0 + (oy + 0.5f) * scale_y - 0.5f;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        float sum = 0.0f;\n"
"        float wsum = 0.0f;\n"
"        int y0 = (int)ceil(src_y - support_y);\n"
"        int y1 = (int)floor(src_y + support_y);\n"
"        int x0 = (int)ceil(src_x - support_x);\n"
"        int x1 = (int)floor(src_x + support_x);\n"
"        for (int j = y0; j <= y1; j++) {\n"
"            float dy = (src_y - j) / fmax(scale_y, 1.0f);\n"
"            float wy = lanczos_weight(dy, a);\n"
"            int sy = clamp(j, 0, in_h - 1);\n"
"            for (int i = x0; i <= x1; i++) {\n"
"                float dx = (src_x - i) / fmax(scale_x, 1.0f);\n"
"                float wx = lanczos_weight(dx, a);\n"
"                int sx = clamp(i, 0, in_w - 1);\n"
"                float w = wx * wy;\n"
"                sum += input[sy * in_linesize + sx * pixelsize + c] * w;\n"
"                wsum += w;\n"
"            }\n"
"        }\n"
"        if (wsum > 0.0f) sum /= wsum;\n"
"        output[oy * out_linesize + ox * pixelsize + c] =\n"
"            (uchar)clamp((int)(sum + 0.5f), 0, 255);\n"
"    }\n"
"}\n"
"\n"
"/* Nearest neighbor */\n"
"__kernel void resample_nearest(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float box_x0, float box_y0, float box_x1, float box_y1)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"\n"
"    float scale_x = (box_x1 - box_x0) / (float)out_w;\n"
"    float scale_y = (box_y1 - box_y0) / (float)out_h;\n"
"    int sx = clamp((int)(box_x0 + ox * scale_x), 0, in_w - 1);\n"
"    int sy = clamp((int)(box_y0 + oy * scale_y), 0, in_h - 1);\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        output[oy * out_linesize + ox * pixelsize + c] =\n"
"            input[sy * in_linesize + sx * pixelsize + c];\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_CONVERT =
"/* RGB to L (grayscale): L = 0.299*R + 0.587*G + 0.114*B */\n"
"__kernel void convert_rgb_to_l(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int si = gid * 4;\n"
"    float r = input[si], g = input[si + 1], b = input[si + 2];\n"
"    output[gid] = (uchar)clamp(\n"
"        (int)(r * 0.299f + g * 0.587f + b * 0.114f + 0.5f), 0, 255);\n"
"}\n"
"\n"
"/* L to RGB: R=G=B=L, A=255 */\n"
"__kernel void convert_l_to_rgb(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    uchar v = input[gid];\n"
"    int oi = gid * 4;\n"
"    output[oi] = v;\n"
"    output[oi + 1] = v;\n"
"    output[oi + 2] = v;\n"
"    output[oi + 3] = 255;\n"
"}\n"
"\n"
"/* RGB to RGBA: copy RGB, set A=255 */\n"
"__kernel void convert_rgb_to_rgba(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int i = gid * 4;\n"
"    output[i] = input[i];\n"
"    output[i + 1] = input[i + 1];\n"
"    output[i + 2] = input[i + 2];\n"
"    output[i + 3] = 255;\n"
"}\n"
"\n"
"/* RGBA to RGB: copy RGB, pad=0 */\n"
"__kernel void convert_rgba_to_rgb(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int i = gid * 4;\n"
"    output[i] = input[i];\n"
"    output[i + 1] = input[i + 1];\n"
"    output[i + 2] = input[i + 2];\n"
"    output[i + 3] = 0;\n"
"}\n"
"\n"
"/* RGBA to L */\n"
"__kernel void convert_rgba_to_l(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int si = gid * 4;\n"
"    float r = input[si], g = input[si + 1], b = input[si + 2];\n"
"    output[gid] = (uchar)clamp(\n"
"        (int)(r * 0.299f + g * 0.587f + b * 0.114f + 0.5f), 0, 255);\n"
"}\n"
"\n"
"/* L to RGBA */\n"
"__kernel void convert_l_to_rgba(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    uchar v = input[gid];\n"
"    int oi = gid * 4;\n"
"    output[oi] = v;\n"
"    output[oi + 1] = v;\n"
"    output[oi + 2] = v;\n"
"    output[oi + 3] = 255;\n"
"}\n"
"\n"
"/* RGB to CMYK */\n"
"__kernel void convert_rgb_to_cmyk(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int i = gid * 4;\n"
"    output[i] = 255 - input[i];\n"
"    output[i + 1] = 255 - input[i + 1];\n"
"    output[i + 2] = 255 - input[i + 2];\n"
"    output[i + 3] = 0;\n"
"}\n"
"\n"
"/* CMYK to RGB */\n"
"__kernel void convert_cmyk_to_rgb(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int i = gid * 4;\n"
"    int c = input[i], m = input[i+1], y_val = input[i+2], k = input[i+3];\n"
"    output[i]   = (uchar)CLIP8(255 - c - k);\n"
"    output[i+1] = (uchar)CLIP8(255 - m - k);\n"
"    output[i+2] = (uchar)CLIP8(255 - y_val - k);\n"
"    output[i+3] = 255;\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_BLEND =
"/* Linear blend: out = im1 * (1-alpha) + im2 * alpha */\n"
"__kernel void blend(\n"
"    __global const uchar *im1,\n"
"    __global const uchar *im2,\n"
"    __global uchar *output,\n"
"    int total_bytes, float alpha)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    float v = (float)im1[i] * (1.0f - alpha) + (float)im2[i] * alpha;\n"
"    output[i] = (uchar)clamp((int)(v + 0.5f), 0, 255);\n"
"}\n"
"\n"
"/* Porter-Duff alpha composite (RGBA only, 4 bytes/pixel) */\n"
"__kernel void alpha_composite(\n"
"    __global const uchar *im1,\n"
"    __global const uchar *im2,\n"
"    __global uchar *output,\n"
"    int num_pixels)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int i = gid * 4;\n"
"\n"
"    float sR = im2[i], sG = im2[i+1], sB = im2[i+2], sA = im2[i+3];\n"
"    float dR = im1[i], dG = im1[i+1], dB = im1[i+2], dA = im1[i+3];\n"
"\n"
"    float sa = sA / 255.0f;\n"
"    float da = dA / 255.0f;\n"
"    float outa = sa + da * (1.0f - sa);\n"
"\n"
"    if (outa < 1e-6f) {\n"
"        output[i] = output[i+1] = output[i+2] = output[i+3] = 0;\n"
"        return;\n"
"    }\n"
"\n"
"    output[i]   = (uchar)clamp((int)((sR * sa + dR * da * (1.0f - sa)) / outa + 0.5f), 0, 255);\n"
"    output[i+1] = (uchar)clamp((int)((sG * sa + dG * da * (1.0f - sa)) / outa + 0.5f), 0, 255);\n"
"    output[i+2] = (uchar)clamp((int)((sB * sa + dB * da * (1.0f - sa)) / outa + 0.5f), 0, 255);\n"
"    output[i+3] = (uchar)clamp((int)(outa * 255.0f + 0.5f), 0, 255);\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_CHOPS =
"/* Channel operations — parameterized by 'op' */\n"
"__kernel void chop_operation(\n"
"    __global const uchar *im1,\n"
"    __global const uchar *im2,\n"
"    __global uchar *output,\n"
"    int total_bytes,\n"
"    int op, float scale, int offset_val)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    int a = im1[i], b = im2[i];\n"
"    int result;\n"
"\n"
"    switch (op) {\n"
"        case 0:  /* add */\n"
"            result = (int)((float)(a + b) / scale + offset_val);\n"
"            break;\n"
"        case 1:  /* subtract */\n"
"            result = (int)((float)(a - b) / scale + offset_val);\n"
"            break;\n"
"        case 2:  /* multiply */\n"
"            result = MULDIV255(a, b);\n"
"            break;\n"
"        case 3:  /* screen */\n"
"            result = 255 - MULDIV255(255 - a, 255 - b);\n"
"            break;\n"
"        case 4:  /* overlay */\n"
"            result = (a < 128)\n"
"                ? MULDIV255(2 * a, b)\n"
"                : 255 - MULDIV255(2 * (255 - a), 255 - b);\n"
"            break;\n"
"        case 5:  /* difference */\n"
"            result = abs(a - b);\n"
"            break;\n"
"        case 6:  /* lighter */\n"
"            result = max(a, b);\n"
"            break;\n"
"        case 7:  /* darker */\n"
"            result = min(a, b);\n"
"            break;\n"
"        case 8:  /* add_modulo */\n"
"            result = (a + b) & 0xFF;\n"
"            break;\n"
"        case 9:  /* subtract_modulo */\n"
"            result = (a - b) & 0xFF;\n"
"            break;\n"
"        case 10: /* soft_light */\n"
"            result = MULDIV255(a, a + MULDIV255(2 * b, 255 - a));\n"
"            break;\n"
"        case 11: /* hard_light */\n"
"            result = (b < 128)\n"
"                ? MULDIV255(2 * b, a)\n"
"                : 255 - MULDIV255(2 * (255 - b), 255 - a);\n"
"            break;\n"
"        case 12: /* and */\n"
"            result = a & b;\n"
"            break;\n"
"        case 13: /* or */\n"
"            result = a | b;\n"
"            break;\n"
"        case 14: /* xor */\n"
"            result = a ^ b;\n"
"            break;\n"
"        case 15: /* invert */\n"
"            result = 255 - a;\n"
"            break;\n"
"        default:\n"
"            result = a;\n"
"    }\n"
"    output[i] = (uchar)clamp(result, 0, 255);\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_GEOMETRY =
"/* Transpose / flip / rotate operations */\n"
"__kernel void transpose_op(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize, int op)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"\n"
"    int sx, sy;\n"
"    switch (op) {\n"
"        case 0: /* FLIP_LEFT_RIGHT */\n"
"            sx = in_w - 1 - ox; sy = oy; break;\n"
"        case 1: /* FLIP_TOP_BOTTOM */\n"
"            sx = ox; sy = in_h - 1 - oy; break;\n"
"        case 2: /* ROTATE_90 */\n"
"            sx = oy; sy = in_h - 1 - ox; break;\n"
"        case 3: /* ROTATE_180 */\n"
"            sx = in_w - 1 - ox; sy = in_h - 1 - oy; break;\n"
"        case 4: /* ROTATE_270 */\n"
"            sx = in_w - 1 - oy; sy = ox; break;\n"
"        case 5: /* TRANSPOSE */\n"
"            sx = oy; sy = ox; break;\n"
"        case 6: /* TRANSVERSE */\n"
"            sx = in_w - 1 - oy; sy = in_h - 1 - ox; break;\n"
"        default:\n"
"            sx = ox; sy = oy;\n"
"    }\n"
"\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"    int src_off = sy * in_linesize + sx * pixelsize;\n"
"    int dst_off = oy * out_linesize + ox * pixelsize;\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        output[dst_off + c] = input[src_off + c];\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_POINT =
"/* Point LUT transform */\n"
"__kernel void point_lut(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const uchar *lut,\n"
"    int total_pixels, int bands)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= total_pixels) return;\n"
"    if (bands == 1) {\n"
"        output[gid] = lut[input[gid]];\n"
"    } else {\n"
"        int base = gid * 4;\n"
"        for (int b = 0; b < bands && b < 4; b++) {\n"
"            output[base + b] = lut[b * 256 + input[base + b]];\n"
"        }\n"
"        /* Preserve padding/alpha if bands < 4 */\n"
"        for (int b = bands; b < 4; b++) {\n"
"            output[base + b] = input[base + b];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Point transform: scale + offset */\n"
"__kernel void point_transform(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int total_bytes,\n"
"    float scale, float offset)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    int val = (int)((float)input[i] * scale + offset + 0.5f);\n"
"    output[i] = (uchar)clamp(val, 0, 255);\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_BANDS =
"/* Extract single band from multi-band image */\n"
"__kernel void getband(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int num_pixels, int pixelsize, int band)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    output[gid] = input[gid * pixelsize + band];\n"
"}\n"
"\n"
"/* Put single band into multi-band image */\n"
"__kernel void putband(\n"
"    __global uchar *image,\n"
"    __global const uchar *band_data,\n"
"    int num_pixels, int pixelsize, int band)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    image[gid * pixelsize + band] = band_data[gid];\n"
"}\n"
"\n"
"/* Fill single band with constant */\n"
"__kernel void fillband(\n"
"    __global uchar *image,\n"
"    int num_pixels, int pixelsize, int band, int color)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    image[gid * pixelsize + band] = (uchar)color;\n"
"}\n"
"\n"
"/* Fill entire image with color */\n"
"__kernel void fill_color(\n"
"    __global uchar *image,\n"
"    int num_pixels, int pixelsize,\n"
"    int c0, int c1, int c2, int c3)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * pixelsize;\n"
"    if (pixelsize >= 1) image[base] = (uchar)c0;\n"
"    if (pixelsize >= 2) image[base + 1] = (uchar)c1;\n"
"    if (pixelsize >= 3) image[base + 2] = (uchar)c2;\n"
"    if (pixelsize >= 4) image[base + 3] = (uchar)c3;\n"
"}\n"
"\n"
"/* Merge bands: combine N single-band images into one multi-band image */\n"
"__kernel void merge_bands(\n"
"    __global const uchar *b0,\n"
"    __global const uchar *b1,\n"
"    __global const uchar *b2,\n"
"    __global const uchar *b3,\n"
"    __global uchar *output,\n"
"    int num_pixels, int nbands)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * 4;\n"
"    output[base] = b0[gid];\n"
"    output[base + 1] = (nbands >= 2) ? b1[gid] : 0;\n"
"    output[base + 2] = (nbands >= 3) ? b2[gid] : 0;\n"
"    output[base + 3] = (nbands >= 4) ? b3[gid] : 255;\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_UNSHARP =
"/* Unsharp mask: sharpen after blur */\n"
"__kernel void unsharp_mask(\n"
"    __global const uchar *original,\n"
"    __global const uchar *blurred,\n"
"    __global uchar *output,\n"
"    int total_bytes, int pixelsize,\n"
"    int percent, int threshold)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    int diff = (int)original[i] - (int)blurred[i];\n"
"    if (abs(diff) >= threshold) {\n"
"        int val = (int)original[i] + diff * percent / 100;\n"
"        output[i] = (uchar)clamp(val, 0, 255);\n"
"    } else {\n"
"        output[i] = original[i];\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_PASTE =
"/* Paste with optional mask */\n"
"__kernel void paste_with_mask(\n"
"    __global uchar *dest,\n"
"    __global const uchar *src,\n"
"    __global const uchar *mask,\n"
"    int dest_w, int dest_h,\n"
"    int src_w, int src_h,\n"
"    int pixelsize,\n"
"    int dx, int dy)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int sy = gid / src_w;\n"
"    int sx = gid % src_w;\n"
"    if (sx >= src_w || sy >= src_h) return;\n"
"    int dest_x = dx + sx;\n"
"    int dest_y = dy + sy;\n"
"    if (dest_x < 0 || dest_x >= dest_w || dest_y < 0 || dest_y >= dest_h) return;\n"
"\n"
"    int alpha = mask[sy * src_w + sx]; /* L mask, 1 byte per pixel */\n"
"    int dest_linesize = dest_w * pixelsize;\n"
"    int src_linesize = src_w * pixelsize;\n"
"\n"
"    for (int c = 0; c < pixelsize; c++) {\n"
"        int si = sy * src_linesize + sx * pixelsize + c;\n"
"        int di = dest_y * dest_linesize + dest_x * pixelsize + c;\n"
"        dest[di] = (uchar)((src[si] * alpha + dest[di] * (255 - alpha) + 128) >> 8);\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_HISTOGRAM =
"/* Per-channel histogram with local-memory reduction */\n"
"__kernel void histogram(\n"
"    __global const uchar *input,\n"
"    __global volatile int *hist,\n"
"    int num_pixels, int pixelsize, int bands)\n"
"{\n"
"    /* Each work-group builds a local histogram, then atomically merges */\n"
"    __local int local_hist[1024]; /* up to 4 bands * 256 */\n"
"    int lid = get_local_id(0);\n"
"    int lsize = get_local_size(0);\n"
"    int total_bins = bands * 256;\n"
"    /* Initialize local histogram to zero */\n"
"    for (int i = lid; i < total_bins; i += lsize) {\n"
"        local_hist[i] = 0;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    /* Each work-item processes one pixel */\n"
"    int gid = get_global_id(0);\n"
"    if (gid < num_pixels) {\n"
"        if (pixelsize == 1) {\n"
"            atomic_add(&local_hist[input[gid]], 1);\n"
"        } else {\n"
"            int base = gid * pixelsize;\n"
"            for (int b = 0; b < bands; b++) {\n"
"                atomic_add(&local_hist[b * 256 + input[base + b]], 1);\n"
"            }\n"
"        }\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    /* Merge local histogram into global using one atomic per bin per workgroup */\n"
"    for (int i = lid; i < total_bins; i += lsize) {\n"
"        if (local_hist[i] > 0) {\n"
"            atomic_add(&hist[i], local_hist[i]);\n"
"        }\n"
"    }\n"
"}\n"
"\n";

static const char *OPENCL_KERNEL_TRANSFORM =
"/* Affine transform */\n"
"__kernel void affine_transform(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float a0, float a1, float a2,\n"
"    float a3, float a4, float a5,\n"
"    int filter_mode, int fill)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int out_linesize = out_w * pixelsize;\n"
"    int in_linesize = in_w * pixelsize;\n"
"\n"
"    float src_xf = a0 * ox + a1 * oy + a2;\n"
"    float src_yf = a3 * ox + a4 * oy + a5;\n"
"\n"
"    if (filter_mode == 0) {\n"
"        /* Nearest */\n"
"        int sx = (int)src_xf;\n"
"        int sy = (int)src_yf;\n"
"        if (sx >= 0 && sx < in_w && sy >= 0 && sy < in_h) {\n"
"            for (int c = 0; c < pixelsize; c++) {\n"
"                output[oy * out_linesize + ox * pixelsize + c] =\n"
"                    input[sy * in_linesize + sx * pixelsize + c];\n"
"            }\n"
"        } else if (fill) {\n"
"            for (int c = 0; c < pixelsize; c++) {\n"
"                output[oy * out_linesize + ox * pixelsize + c] = 0;\n"
"            }\n"
"        }\n"
"    } else {\n"
"        /* Bilinear */\n"
"        int x0 = (int)floor(src_xf);\n"
"        int y0 = (int)floor(src_yf);\n"
"        float fx = src_xf - x0;\n"
"        float fy = src_yf - y0;\n"
"        if (x0 >= 0 && x0 + 1 < in_w && y0 >= 0 && y0 + 1 < in_h) {\n"
"            for (int c = 0; c < pixelsize; c++) {\n"
"                float v00 = input[y0 * in_linesize + x0 * pixelsize + c];\n"
"                float v10 = input[y0 * in_linesize + (x0+1) * pixelsize + c];\n"
"                float v01 = input[(y0+1) * in_linesize + x0 * pixelsize + c];\n"
"                float v11 = input[(y0+1) * in_linesize + (x0+1) * pixelsize + c];\n"
"                float top = v00 + (v10 - v00) * fx;\n"
"                float bot = v01 + (v11 - v01) * fx;\n"
"                float val = top + (bot - top) * fy;\n"
"                output[oy * out_linesize + ox * pixelsize + c] =\n"
"                    (uchar)clamp((int)(val + 0.5f), 0, 255);\n"
"            }\n"
"        } else if (fill) {\n"
"            for (int c = 0; c < pixelsize; c++) {\n"
"                output[oy * out_linesize + ox * pixelsize + c] = 0;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Perspective transform */\n"
"__kernel void perspective_transform(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize,\n"
"    float a0, float a1, float a2,\n"
"    float a3, float a4, float a5,\n"
"    float a6, float a7,\n"
"    int filter_mode, int fill)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    if (ox >= out_w || oy >= out_h) return;\n"
"    int out_linesize = out_w * pixelsize;\n"
"    int in_linesize = in_w * pixelsize;\n"
"\n"
"    float w = a6 * ox + a7 * oy + 1.0f;\n"
"    if (fabs(w) < 1e-10f) w = 1.0f;\n"
"    float src_xf = (a0 * ox + a1 * oy + a2) / w;\n"
"    float src_yf = (a3 * ox + a4 * oy + a5) / w;\n"
"\n"
"    /* Nearest neighbor for perspective */\n"
"    int sx = (int)src_xf;\n"
"    int sy = (int)src_yf;\n"
"    if (sx >= 0 && sx < in_w && sy >= 0 && sy < in_h) {\n"
"        for (int c = 0; c < pixelsize; c++) {\n"
"            output[oy * out_linesize + ox * pixelsize + c] =\n"
"                input[sy * in_linesize + sx * pixelsize + c];\n"
"        }\n"
"    } else if (fill) {\n"
"        for (int c = 0; c < pixelsize; c++) {\n"
"            output[oy * out_linesize + ox * pixelsize + c] = 0;\n"
"        }\n"
"    }\n"
"}\n"
"\n";

/* ================================================================== */
/* OpenCL context state                                                 */
/* ================================================================== */

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program program;

    /* Compiled kernel handles */
    cl_kernel k_box_blur_h;
    cl_kernel k_box_blur_v;
    cl_kernel k_gaussian_blur_h;
    cl_kernel k_gaussian_blur_v;
    cl_kernel k_convolve;
    cl_kernel k_resample_nearest;
    cl_kernel k_resample_bilinear;
    cl_kernel k_resample_bicubic;
    cl_kernel k_resample_lanczos;
    cl_kernel k_convert_rgb_to_l;
    cl_kernel k_convert_l_to_rgb;
    cl_kernel k_convert_rgb_to_rgba;
    cl_kernel k_convert_rgba_to_rgb;
    cl_kernel k_convert_rgba_to_l;
    cl_kernel k_convert_l_to_rgba;
    cl_kernel k_convert_rgb_to_cmyk;
    cl_kernel k_convert_cmyk_to_rgb;
    cl_kernel k_blend;
    cl_kernel k_alpha_composite;
    cl_kernel k_chop_operation;
    cl_kernel k_transpose_op;
    cl_kernel k_point_lut;
    cl_kernel k_point_transform;
    cl_kernel k_getband;
    cl_kernel k_putband;
    cl_kernel k_fillband;
    cl_kernel k_fill_color;
    cl_kernel k_merge_bands;
    cl_kernel k_unsharp_mask;
    cl_kernel k_paste_with_mask;
    cl_kernel k_histogram;
    cl_kernel k_affine_transform;
    cl_kernel k_perspective_transform;
    /* New kernels */
    cl_kernel k_color_matrix;
    cl_kernel k_color_lut_3d;
    cl_kernel k_crop_region;
    cl_kernel k_expand_image;
    cl_kernel k_offset_image;
    cl_kernel k_linear_gradient;
    cl_kernel k_radial_gradient;
    cl_kernel k_negative;
    cl_kernel k_posterize;
    cl_kernel k_solarize;
    cl_kernel k_equalize;
    cl_kernel k_getbbox_kernel;
    cl_kernel k_getextrema_kernel;
    cl_kernel k_effect_spread_kernel;
    cl_kernel k_convert_ycbcr_to_rgb;
    cl_kernel k_convert_rgb_to_ycbcr;
    cl_kernel k_reduce;
    cl_kernel k_rank_filter;
    cl_kernel k_mode_filter;
} OpenCLContext;

/* ================================================================== */
/* Helpers                                                              */
/* ================================================================== */

static cl_kernel
_ocl_create_kernel(cl_program program, const char *name) {
    cl_int err;
    cl_kernel k = clCreateKernel(program, name, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Pillow GPU: failed to create kernel '%s' (err=%d)\n",
                name, err);
        return NULL;
    }
    return k;
}

static size_t
_ocl_round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/* ================================================================== */
/* Backend operation implementations                                    */
/* ================================================================== */

static void
_ocl_shutdown(GPUBackend self) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx) return;

    /* Release all kernels */
    cl_kernel *kernels = (cl_kernel *)&ctx->k_box_blur_h;
    size_t num_kernels = (sizeof(OpenCLContext) - offsetof(OpenCLContext, k_box_blur_h))
                         / sizeof(cl_kernel);
    for (size_t i = 0; i < num_kernels; i++) {
        if (kernels[i]) clReleaseKernel(kernels[i]);
    }

    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);

    free(ctx);
    free(self);
}

static int
_ocl_buffer_alloc(GPUBackend self, GPUBuffer buf, size_t size) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    cl_int err;
    buf->handle.cl_mem = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE,
                                        size, NULL, &err);
    if (err != CL_SUCCESS) return GPU_ERROR_MEMORY;
    buf->size = size;
    return GPU_OK;
}

static void
_ocl_buffer_free(GPUBackend self, GPUBuffer buf) {
    (void)self;
    if (buf->handle.cl_mem) {
        clReleaseMemObject((cl_mem)buf->handle.cl_mem);
        buf->handle.cl_mem = NULL;
        buf->size = 0;
    }
}

static int
_ocl_buffer_upload(GPUBackend self, GPUBuffer buf, const void *data, size_t size) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    cl_int err = clEnqueueWriteBuffer(ctx->queue, (cl_mem)buf->handle.cl_mem,
                                       CL_TRUE, 0, size, data, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? GPU_OK : GPU_ERROR_TRANSFER;
}

static int
_ocl_buffer_download(GPUBackend self, GPUBuffer buf, void *data, size_t size) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    cl_int err = clEnqueueReadBuffer(ctx->queue, (cl_mem)buf->handle.cl_mem,
                                      CL_TRUE, 0, size, data, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? GPU_OK : GPU_ERROR_TRANSFER;
}

static int
_ocl_buffer_copy(GPUBackend self, GPUBuffer dst, GPUBuffer src, size_t size) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    cl_int err = clEnqueueCopyBuffer(ctx->queue, (cl_mem)src->handle.cl_mem,
                                      (cl_mem)dst->handle.cl_mem,
                                      0, 0, size, 0, NULL, NULL);
    if (err != CL_SUCCESS) return GPU_ERROR_TRANSFER;
    clFinish(ctx->queue);
    return GPU_OK;
}

/* -------------------------------------------------------------------- */
/* Blur operations                                                       */
/* -------------------------------------------------------------------- */

static int
_ocl_box_blur(GPUBackend self, ImagingGPU out, ImagingGPU in,
              float xradius, float yradius, int n) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int rx = (int)(xradius + 0.5f);
    int ry = (int)(yradius + 0.5f);
    if (rx < 0) rx = 0;
    if (ry < 0) ry = 0;

    size_t buf_size = in->buffer.size;

    /* We need a temp buffer for ping-pong */
    GPUBufferInstance tmp_buf = {0};
    int err = _ocl_buffer_alloc(self, &tmp_buf, buf_size);
    if (err != GPU_OK) return err;

    /* Copy input to out initially */
    _ocl_buffer_copy(self, &out->buffer, &in->buffer, buf_size);

    cl_int clerr;
    for (int pass = 0; pass < n; pass++) {
        /* Horizontal pass: out -> tmp */
        clerr = clSetKernelArg(ctx->k_box_blur_h, 0, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clerr |= clSetKernelArg(ctx->k_box_blur_h, 1, sizeof(cl_mem), &tmp_buf.handle.cl_mem);
        clerr |= clSetKernelArg(ctx->k_box_blur_h, 2, sizeof(int), &in->xsize);
        clerr |= clSetKernelArg(ctx->k_box_blur_h, 3, sizeof(int), &in->ysize);
        clerr |= clSetKernelArg(ctx->k_box_blur_h, 4, sizeof(int), &in->pixelsize);
        clerr |= clSetKernelArg(ctx->k_box_blur_h, 5, sizeof(int), &rx);
        if (clerr != CL_SUCCESS) { _ocl_buffer_free(self, &tmp_buf); return GPU_ERROR_LAUNCH; }

        size_t total_pixels = (size_t)in->xsize * in->ysize;
        size_t global_h = _ocl_round_up(total_pixels, 256);
        clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_box_blur_h, 1, NULL,
                                        &global_h, NULL, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) { _ocl_buffer_free(self, &tmp_buf); return GPU_ERROR_LAUNCH; }

        /* Vertical pass: tmp -> out */
        clerr = clSetKernelArg(ctx->k_box_blur_v, 0, sizeof(cl_mem), &tmp_buf.handle.cl_mem);
        clerr |= clSetKernelArg(ctx->k_box_blur_v, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clerr |= clSetKernelArg(ctx->k_box_blur_v, 2, sizeof(int), &in->xsize);
        clerr |= clSetKernelArg(ctx->k_box_blur_v, 3, sizeof(int), &in->ysize);
        clerr |= clSetKernelArg(ctx->k_box_blur_v, 4, sizeof(int), &in->pixelsize);
        clerr |= clSetKernelArg(ctx->k_box_blur_v, 5, sizeof(int), &ry);
        if (clerr != CL_SUCCESS) { _ocl_buffer_free(self, &tmp_buf); return GPU_ERROR_LAUNCH; }

        size_t global_v = _ocl_round_up(total_pixels, 256);
        clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_box_blur_v, 1, NULL,
                                        &global_v, NULL, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) { _ocl_buffer_free(self, &tmp_buf); return GPU_ERROR_LAUNCH; }
    }

    clFinish(ctx->queue);
    _ocl_buffer_free(self, &tmp_buf);
    return GPU_OK;
}

static int
_ocl_gaussian_blur(GPUBackend self, ImagingGPU out, ImagingGPU in,
                   float xradius, float yradius, int passes) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;

    /* Compute Gaussian weights */
    int rx = (int)ceilf(xradius * 2.57f); /* 2.57 sigma covers 99.5% */
    int ry = (int)ceilf(yradius * 2.57f);
    if (rx < 1) rx = 1;
    if (ry < 1) ry = 1;
    int xdiam = 2 * rx + 1;
    int ydiam = 2 * ry + 1;

    float *xweights = (float *)malloc(xdiam * sizeof(float));
    float *yweights = (float *)malloc(ydiam * sizeof(float));
    if (!xweights || !yweights) {
        free(xweights); free(yweights);
        return GPU_ERROR_MEMORY;
    }

    /* Generate Gaussian kernel */
    float xsum = 0, ysum = 0;
    for (int i = 0; i < xdiam; i++) {
        float d = (float)(i - rx);
        xweights[i] = expf(-(d * d) / (2.0f * xradius * xradius));
        xsum += xweights[i];
    }
    for (int i = 0; i < ydiam; i++) {
        float d = (float)(i - ry);
        yweights[i] = expf(-(d * d) / (2.0f * yradius * yradius));
        ysum += yweights[i];
    }
    for (int i = 0; i < xdiam; i++) xweights[i] /= xsum;
    for (int i = 0; i < ydiam; i++) yweights[i] /= ysum;

    /* Upload weights to GPU */
    cl_int clerr;
    cl_mem xw_buf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    xdiam * sizeof(float), xweights, &clerr);
    cl_mem yw_buf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    ydiam * sizeof(float), yweights, &clerr);
    free(xweights);
    free(yweights);
    if (!xw_buf || !yw_buf) {
        if (xw_buf) clReleaseMemObject(xw_buf);
        if (yw_buf) clReleaseMemObject(yw_buf);
        return GPU_ERROR_MEMORY;
    }

    /* Temp buffer for ping-pong */
    GPUBufferInstance tmp_buf = {0};
    int err = _ocl_buffer_alloc(self, &tmp_buf, in->buffer.size);
    if (err != GPU_OK) {
        clReleaseMemObject(xw_buf);
        clReleaseMemObject(yw_buf);
        return err;
    }

    size_t total_pixels = (size_t)in->xsize * in->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);

    _ocl_buffer_copy(self, &out->buffer, &in->buffer, in->buffer.size);

    for (int pass = 0; pass < passes; pass++) {
        /* Horizontal: out -> tmp */
        clSetKernelArg(ctx->k_gaussian_blur_h, 0, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(ctx->k_gaussian_blur_h, 1, sizeof(cl_mem), &tmp_buf.handle.cl_mem);
        clSetKernelArg(ctx->k_gaussian_blur_h, 2, sizeof(cl_mem), &xw_buf);
        clSetKernelArg(ctx->k_gaussian_blur_h, 3, sizeof(int), &in->xsize);
        clSetKernelArg(ctx->k_gaussian_blur_h, 4, sizeof(int), &in->ysize);
        clSetKernelArg(ctx->k_gaussian_blur_h, 5, sizeof(int), &in->pixelsize);
        clSetKernelArg(ctx->k_gaussian_blur_h, 6, sizeof(int), &rx);
        clEnqueueNDRangeKernel(ctx->queue, ctx->k_gaussian_blur_h, 1, NULL,
                                &global, NULL, 0, NULL, NULL);

        /* Vertical: tmp -> out */
        clSetKernelArg(ctx->k_gaussian_blur_v, 0, sizeof(cl_mem), &tmp_buf.handle.cl_mem);
        clSetKernelArg(ctx->k_gaussian_blur_v, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(ctx->k_gaussian_blur_v, 2, sizeof(cl_mem), &yw_buf);
        clSetKernelArg(ctx->k_gaussian_blur_v, 3, sizeof(int), &in->xsize);
        clSetKernelArg(ctx->k_gaussian_blur_v, 4, sizeof(int), &in->ysize);
        clSetKernelArg(ctx->k_gaussian_blur_v, 5, sizeof(int), &in->pixelsize);
        clSetKernelArg(ctx->k_gaussian_blur_v, 6, sizeof(int), &ry);
        clEnqueueNDRangeKernel(ctx->queue, ctx->k_gaussian_blur_v, 1, NULL,
                                &global, NULL, 0, NULL, NULL);
    }

    clFinish(ctx->queue);
    clReleaseMemObject(xw_buf);
    clReleaseMemObject(yw_buf);
    _ocl_buffer_free(self, &tmp_buf);
    return GPU_OK;
}

static int
_ocl_unsharp_mask(GPUBackend self, ImagingGPU out, ImagingGPU in,
                  float radius, int percent, int threshold) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;

    /* First blur the image */
    ImagingGPU blurred = ImagingGPU_NewDirty(in->mode, in->xsize, in->ysize);
    if (!blurred) return GPU_ERROR_MEMORY;

    int err = _ocl_gaussian_blur(self, blurred, in, radius, radius, 3);
    if (err != GPU_OK) {
        ImagingGPU_Delete(blurred);
        return err;
    }

    /* Apply unsharp mask */
    int total_bytes = (int)in->buffer.size;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_unsharp_mask, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_unsharp_mask, 1, sizeof(cl_mem), &blurred->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_unsharp_mask, 2, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_unsharp_mask, 3, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_unsharp_mask, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_unsharp_mask, 5, sizeof(int), &percent);
    clSetKernelArg(ctx->k_unsharp_mask, 6, sizeof(int), &threshold);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_unsharp_mask, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    ImagingGPU_Delete(blurred);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Filter (convolution)                                                  */
/* -------------------------------------------------------------------- */

static int
_ocl_filter(GPUBackend self, ImagingGPU out, ImagingGPU in,
            int ksize_x, int ksize_y, const float *kernel,
            float divisor, float offset) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;

    /* Upload kernel to GPU */
    cl_int clerr;
    size_t ksize = ksize_x * ksize_y * sizeof(float);
    cl_mem k_buf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   ksize, (void *)kernel, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    size_t total_pixels = (size_t)in->xsize * in->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);

    clSetKernelArg(ctx->k_convolve, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_convolve, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_convolve, 2, sizeof(cl_mem), &k_buf);
    clSetKernelArg(ctx->k_convolve, 3, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_convolve, 4, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_convolve, 5, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_convolve, 6, sizeof(int), &ksize_x);
    clSetKernelArg(ctx->k_convolve, 7, sizeof(int), &ksize_y);
    clSetKernelArg(ctx->k_convolve, 8, sizeof(float), &divisor);
    clSetKernelArg(ctx->k_convolve, 9, sizeof(float), &offset);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_convolve, 1, NULL,
                                    &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    clReleaseMemObject(k_buf);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Resample                                                              */
/* -------------------------------------------------------------------- */

static int
_ocl_resample(GPUBackend self, ImagingGPU out, ImagingGPU in,
              int xsize, int ysize, int filter, const float box[4]) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;

    cl_kernel k;
    switch (filter) {
        case GPU_RESAMPLE_NEAREST: k = ctx->k_resample_nearest; break;
        case GPU_RESAMPLE_BILINEAR: k = ctx->k_resample_bilinear; break;
        case GPU_RESAMPLE_BICUBIC: k = ctx->k_resample_bicubic; break;
        case GPU_RESAMPLE_LANCZOS: k = ctx->k_resample_lanczos; break;
        default: k = ctx->k_resample_bilinear;
    }

    float bx0 = box[0], by0 = box[1], bx1 = box[2], by1 = box[3];
    size_t total_pixels = (size_t)xsize * ysize;
    size_t global = _ocl_round_up(total_pixels, 256);

    clSetKernelArg(k, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(k, 2, sizeof(int), &in->xsize);
    clSetKernelArg(k, 3, sizeof(int), &in->ysize);
    clSetKernelArg(k, 4, sizeof(int), &xsize);
    clSetKernelArg(k, 5, sizeof(int), &ysize);
    clSetKernelArg(k, 6, sizeof(int), &in->pixelsize);
    clSetKernelArg(k, 7, sizeof(float), &bx0);
    clSetKernelArg(k, 8, sizeof(float), &by0);
    clSetKernelArg(k, 9, sizeof(float), &bx1);
    clSetKernelArg(k, 10, sizeof(float), &by1);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Convert                                                               */
/* -------------------------------------------------------------------- */

static int
_ocl_convert(GPUBackend self, ImagingGPU out, ImagingGPU in, ModeID to_mode) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);
    cl_kernel k = NULL;

    /* Select appropriate conversion kernel */
    if (in->mode == IMAGING_MODE_RGB && to_mode == IMAGING_MODE_L) {
        k = ctx->k_convert_rgb_to_l;
    } else if (in->mode == IMAGING_MODE_L && to_mode == IMAGING_MODE_RGB) {
        k = ctx->k_convert_l_to_rgb;
    } else if (in->mode == IMAGING_MODE_RGB && to_mode == IMAGING_MODE_RGBA) {
        k = ctx->k_convert_rgb_to_rgba;
    } else if (in->mode == IMAGING_MODE_RGBA && to_mode == IMAGING_MODE_RGB) {
        k = ctx->k_convert_rgba_to_rgb;
    } else if (in->mode == IMAGING_MODE_RGBA && to_mode == IMAGING_MODE_L) {
        k = ctx->k_convert_rgba_to_l;
    } else if (in->mode == IMAGING_MODE_L && to_mode == IMAGING_MODE_RGBA) {
        k = ctx->k_convert_l_to_rgba;
    } else if (in->mode == IMAGING_MODE_RGB && to_mode == IMAGING_MODE_CMYK) {
        k = ctx->k_convert_rgb_to_cmyk;
    } else if (in->mode == IMAGING_MODE_CMYK && to_mode == IMAGING_MODE_RGB) {
        k = ctx->k_convert_cmyk_to_rgb;
    } else if ((in->mode == IMAGING_MODE_YCbCr) &&
               (to_mode == IMAGING_MODE_RGB || to_mode == IMAGING_MODE_RGBA)) {
        /* YCbCr -> RGB(A) uses separate kernel with different args */
        cl_kernel kk = ctx->k_convert_ycbcr_to_rgb;
        if (!kk) return GPU_ERROR_UNSUPPORTED;
        int num_pixels = total_pixels;
        int in_ps = in->pixelsize;
        int out_ps = out->pixelsize;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(kk, 2, sizeof(int), &num_pixels);
        clSetKernelArg(kk, 3, sizeof(int), &in_ps);
        clSetKernelArg(kk, 4, sizeof(int), &out_ps);
        cl_int clerr2 = clEnqueueNDRangeKernel(ctx->queue, kk, 1, NULL,
                                                &global, NULL, 0, NULL, NULL);
        clFinish(ctx->queue);
        return (clerr2 == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
    } else if ((in->mode == IMAGING_MODE_RGB || in->mode == IMAGING_MODE_RGBA) &&
               to_mode == IMAGING_MODE_YCbCr) {
        /* RGB(A) -> YCbCr uses separate kernel with different args */
        cl_kernel kk = ctx->k_convert_rgb_to_ycbcr;
        if (!kk) return GPU_ERROR_UNSUPPORTED;
        int num_pixels = total_pixels;
        int in_ps = in->pixelsize;
        int out_ps = out->pixelsize;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(kk, 2, sizeof(int), &num_pixels);
        clSetKernelArg(kk, 3, sizeof(int), &in_ps);
        clSetKernelArg(kk, 4, sizeof(int), &out_ps);
        cl_int clerr2 = clEnqueueNDRangeKernel(ctx->queue, kk, 1, NULL,
                                                &global, NULL, 0, NULL, NULL);
        clFinish(ctx->queue);
        return (clerr2 == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
    } else {
        return GPU_ERROR_UNSUPPORTED;
    }

    clSetKernelArg(k, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(k, 2, sizeof(int), &in->xsize);
    clSetKernelArg(k, 3, sizeof(int), &in->ysize);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Blend / composite                                                     */
/* -------------------------------------------------------------------- */

static int
_ocl_blend(GPUBackend self, ImagingGPU out,
           ImagingGPU im1, ImagingGPU im2, float alpha) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_bytes = (int)im1->buffer.size;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_blend, 0, sizeof(cl_mem), &im1->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_blend, 1, sizeof(cl_mem), &im2->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_blend, 2, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_blend, 3, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_blend, 4, sizeof(float), &alpha);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_blend, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_alpha_composite(GPUBackend self, ImagingGPU out,
                     ImagingGPU im1, ImagingGPU im2) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = im1->xsize * im1->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_alpha_composite, 0, sizeof(cl_mem), &im1->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_alpha_composite, 1, sizeof(cl_mem), &im2->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_alpha_composite, 2, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_alpha_composite, 3, sizeof(int), &num_pixels);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_alpha_composite, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Channel operations                                                    */
/* -------------------------------------------------------------------- */

static int
_ocl_chop(GPUBackend self, ImagingGPU out,
          ImagingGPU im1, ImagingGPU im2,
          int op, float scale, int offset) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_bytes = (int)im1->buffer.size;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_chop_operation, 0, sizeof(cl_mem), &im1->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_chop_operation, 1, sizeof(cl_mem), &im2->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_chop_operation, 2, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_chop_operation, 3, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_chop_operation, 4, sizeof(int), &op);
    clSetKernelArg(ctx->k_chop_operation, 5, sizeof(float), &scale);
    clSetKernelArg(ctx->k_chop_operation, 6, sizeof(int), &offset);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_chop_operation, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Geometry                                                              */
/* -------------------------------------------------------------------- */

static int
_ocl_transpose(GPUBackend self, ImagingGPU out, ImagingGPU in, int op) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);

    clSetKernelArg(ctx->k_transpose_op, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_transpose_op, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_transpose_op, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_transpose_op, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_transpose_op, 4, sizeof(int), &out->xsize);
    clSetKernelArg(ctx->k_transpose_op, 5, sizeof(int), &out->ysize);
    clSetKernelArg(ctx->k_transpose_op, 6, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_transpose_op, 7, sizeof(int), &op);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_transpose_op, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_transform(GPUBackend self, ImagingGPU out, ImagingGPU in,
               int method, double a[8], int filter, int fill) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);
    cl_kernel k;

    float fa[8];
    for (int i = 0; i < 8; i++) fa[i] = (float)a[i];

    if (method == 2) { /* PERSPECTIVE */
        k = ctx->k_perspective_transform;
        clSetKernelArg(k, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
        clSetKernelArg(k, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(k, 2, sizeof(int), &in->xsize);
        clSetKernelArg(k, 3, sizeof(int), &in->ysize);
        clSetKernelArg(k, 4, sizeof(int), &out->xsize);
        clSetKernelArg(k, 5, sizeof(int), &out->ysize);
        clSetKernelArg(k, 6, sizeof(int), &in->pixelsize);
        for (int i = 0; i < 8; i++)
            clSetKernelArg(k, 7 + i, sizeof(float), &fa[i]);
        clSetKernelArg(k, 15, sizeof(int), &filter);
        clSetKernelArg(k, 16, sizeof(int), &fill);
    } else { /* AFFINE */
        k = ctx->k_affine_transform;
        clSetKernelArg(k, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
        clSetKernelArg(k, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
        clSetKernelArg(k, 2, sizeof(int), &in->xsize);
        clSetKernelArg(k, 3, sizeof(int), &in->ysize);
        clSetKernelArg(k, 4, sizeof(int), &out->xsize);
        clSetKernelArg(k, 5, sizeof(int), &out->ysize);
        clSetKernelArg(k, 6, sizeof(int), &in->pixelsize);
        for (int i = 0; i < 6; i++)
            clSetKernelArg(k, 7 + i, sizeof(float), &fa[i]);
        clSetKernelArg(k, 13, sizeof(int), &filter);
        clSetKernelArg(k, 14, sizeof(int), &fill);
    }

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Point / LUT                                                           */
/* -------------------------------------------------------------------- */

static int
_ocl_point_lut(GPUBackend self, ImagingGPU out, ImagingGPU in,
               const UINT8 *lut, int bands) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int lut_size = bands * 256;

    cl_int clerr;
    cl_mem lut_buf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     lut_size, (void *)lut, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    int total_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(total_pixels, 256);

    clSetKernelArg(ctx->k_point_lut, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_point_lut, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_point_lut, 2, sizeof(cl_mem), &lut_buf);
    clSetKernelArg(ctx->k_point_lut, 3, sizeof(int), &total_pixels);
    clSetKernelArg(ctx->k_point_lut, 4, sizeof(int), &bands);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_point_lut, 1, NULL,
                                    &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    clReleaseMemObject(lut_buf);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_point_transform(GPUBackend self, ImagingGPU out, ImagingGPU in,
                     double scale, double offset) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int total_bytes = (int)in->buffer.size;
    size_t global = _ocl_round_up(total_bytes, 256);
    float fscale = (float)scale, foffset = (float)offset;

    clSetKernelArg(ctx->k_point_transform, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_point_transform, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_point_transform, 2, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_point_transform, 3, sizeof(float), &fscale);
    clSetKernelArg(ctx->k_point_transform, 4, sizeof(float), &foffset);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_point_transform, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Band operations                                                       */
/* -------------------------------------------------------------------- */

static int
_ocl_getband(GPUBackend self, ImagingGPU out, ImagingGPU in, int band) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_getband, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_getband, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_getband, 2, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_getband, 3, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_getband, 4, sizeof(int), &band);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_getband, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_putband(GPUBackend self, ImagingGPU im, ImagingGPU band_im, int band) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = im->xsize * im->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_putband, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_putband, 1, sizeof(cl_mem), &band_im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_putband, 2, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_putband, 3, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_putband, 4, sizeof(int), &band);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_putband, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_fillband(GPUBackend self, ImagingGPU im, int band, int color) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = im->xsize * im->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_fillband, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_fillband, 1, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_fillband, 2, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_fillband, 3, sizeof(int), &band);
    clSetKernelArg(ctx->k_fillband, 4, sizeof(int), &color);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_fillband, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_fill(GPUBackend self, ImagingGPU im, const void *color) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = im->xsize * im->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    const UINT8 *c = (const UINT8 *)color;
    int c0 = c[0], c1 = (im->pixelsize >= 2) ? c[1] : 0;
    int c2 = (im->pixelsize >= 3) ? c[2] : 0;
    int c3 = (im->pixelsize >= 4) ? c[3] : 0;

    clSetKernelArg(ctx->k_fill_color, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_fill_color, 1, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_fill_color, 2, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_fill_color, 3, sizeof(int), &c0);
    clSetKernelArg(ctx->k_fill_color, 4, sizeof(int), &c1);
    clSetKernelArg(ctx->k_fill_color, 5, sizeof(int), &c2);
    clSetKernelArg(ctx->k_fill_color, 6, sizeof(int), &c3);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_fill_color, 1, NULL,
                                           &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_copy(GPUBackend self, ImagingGPU out, ImagingGPU in) {
    return _ocl_buffer_copy(self, &out->buffer, &in->buffer, in->buffer.size);
}

static int
_ocl_histogram(GPUBackend self, ImagingGPU im, long *hist_out) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    int num_pixels = im->xsize * im->ysize;
    int bands = im->bands;
    int hist_size = bands * 256;

    /* Zero out host hist */
    memset(hist_out, 0, hist_size * sizeof(long));

    /* Create GPU histogram buffer (int, zeroed) */
    cl_int clerr;
    int *zeros = (int *)calloc(hist_size, sizeof(int));
    cl_mem hist_buf = clCreateBuffer(ctx->context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      hist_size * sizeof(int), zeros, &clerr);
    free(zeros);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    size_t global = _ocl_round_up(num_pixels, 256);
    clSetKernelArg(ctx->k_histogram, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_histogram, 1, sizeof(cl_mem), &hist_buf);
    clSetKernelArg(ctx->k_histogram, 2, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_histogram, 3, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_histogram, 4, sizeof(int), &bands);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_histogram, 1, NULL,
                                    &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    /* Read back */
    int *hist_int = (int *)malloc(hist_size * sizeof(int));
    clEnqueueReadBuffer(ctx->queue, hist_buf, CL_TRUE, 0,
                        hist_size * sizeof(int), hist_int, 0, NULL, NULL);
    for (int i = 0; i < hist_size; i++) {
        hist_out[i] = hist_int[i];
    }

    free(hist_int);
    clReleaseMemObject(hist_buf);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* paste: use paste_with_mask kernel                                    */
/* ================================================================== */

static int
_ocl_paste(GPUBackend self, ImagingGPU dest, ImagingGPU src,
           ImagingGPU mask, int dx, int dy, int sx, int sy,
           int sw, int sh) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_paste_with_mask) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = sw * sh;
    if (num_pixels <= 0) return GPU_OK;
    size_t global = _ocl_round_up(num_pixels, 256);

    /* If no mask, create an opaque temp mask */
    cl_mem mask_mem;
    int free_mask = 0;
    if (mask && mask->buffer.handle.cl_mem) {
        mask_mem = mask->buffer.handle.cl_mem;
    } else {
        cl_int clerr2;
        unsigned char *ones = (unsigned char *)malloc(num_pixels);
        if (!ones) return GPU_ERROR_MEMORY;
        memset(ones, 255, num_pixels);
        mask_mem = clCreateBuffer(ctx->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   num_pixels, ones, &clerr2);
        free(ones);
        if (clerr2 != CL_SUCCESS) return GPU_ERROR_MEMORY;
        free_mask = 1;
    }

    clSetKernelArg(ctx->k_paste_with_mask, 0, sizeof(cl_mem), &dest->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_paste_with_mask, 1, sizeof(cl_mem), &src->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_paste_with_mask, 2, sizeof(cl_mem), &mask_mem);
    clSetKernelArg(ctx->k_paste_with_mask, 3, sizeof(int), &dest->xsize);
    clSetKernelArg(ctx->k_paste_with_mask, 4, sizeof(int), &dest->ysize);
    clSetKernelArg(ctx->k_paste_with_mask, 5, sizeof(int), &sw);
    clSetKernelArg(ctx->k_paste_with_mask, 6, sizeof(int), &sh);
    clSetKernelArg(ctx->k_paste_with_mask, 7, sizeof(int), &dest->pixelsize);
    clSetKernelArg(ctx->k_paste_with_mask, 8, sizeof(int), &dx);
    clSetKernelArg(ctx->k_paste_with_mask, 9, sizeof(int), &dy);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_paste_with_mask, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);

    if (free_mask) clReleaseMemObject(mask_mem);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* merge: combine N single-band images into one multi-band image       */
/* ================================================================== */

static int
_ocl_merge(GPUBackend self, ImagingGPU out, ImagingGPU bands[4], int nbands) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_merge_bands) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    cl_mem b0 = bands[0]->buffer.handle.cl_mem;
    cl_mem b1 = (nbands >= 2) ? bands[1]->buffer.handle.cl_mem : b0;
    cl_mem b2 = (nbands >= 3) ? bands[2]->buffer.handle.cl_mem : b0;
    cl_mem b3 = (nbands >= 4) ? bands[3]->buffer.handle.cl_mem : b0;

    clSetKernelArg(ctx->k_merge_bands, 0, sizeof(cl_mem), &b0);
    clSetKernelArg(ctx->k_merge_bands, 1, sizeof(cl_mem), &b1);
    clSetKernelArg(ctx->k_merge_bands, 2, sizeof(cl_mem), &b2);
    clSetKernelArg(ctx->k_merge_bands, 3, sizeof(cl_mem), &b3);
    clSetKernelArg(ctx->k_merge_bands, 4, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_merge_bands, 5, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_merge_bands, 6, sizeof(int), &nbands);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_merge_bands, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* split: extract bands using getband                                  */
/* ================================================================== */

static int
_ocl_split(GPUBackend self, ImagingGPU im, ImagingGPU bands[4]) {
    for (int i = 0; i < im->bands; i++) {
        int err = _ocl_getband(self, bands[i], im, i);
        if (err != GPU_OK) return err;
    }
    return GPU_OK;
}

/* ================================================================== */
/* getbbox: bounding box of non-zero pixels (GPU parallel reduction)   */
/* ================================================================== */

static int
_ocl_getbbox(GPUBackend self, ImagingGPU im, int bbox[4], int alpha_only) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_getbbox_kernel) return GPU_ERROR_UNSUPPORTED;

    /* Allocate result buffer: [min_x, min_y, max_x, max_y] */
    int init_vals[4] = { im->xsize, im->ysize, -1, -1 };
    cl_int clerr;
    cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        4 * sizeof(int), init_vals, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    int num_pixels = im->xsize * im->ysize;
    int check_ch = alpha_only ? (im->pixelsize - 1) : -1;
    size_t global = _ocl_round_up(num_pixels, 256);
    size_t local = 256;

    clSetKernelArg(ctx->k_getbbox_kernel, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_getbbox_kernel, 1, sizeof(cl_mem), &result_buf);
    clSetKernelArg(ctx->k_getbbox_kernel, 2, sizeof(int), &im->xsize);
    clSetKernelArg(ctx->k_getbbox_kernel, 3, sizeof(int), &im->ysize);
    clSetKernelArg(ctx->k_getbbox_kernel, 4, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_getbbox_kernel, 5, sizeof(int), &check_ch);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_getbbox_kernel,
                                    1, NULL, &global, &local, 0, NULL, NULL);
    if (clerr != CL_SUCCESS) { clReleaseMemObject(result_buf); return GPU_ERROR_LAUNCH; }

    int results[4];
    clEnqueueReadBuffer(ctx->queue, result_buf, CL_TRUE, 0,
                        4 * sizeof(int), results, 0, NULL, NULL);
    clReleaseMemObject(result_buf);

    if (results[2] < results[0]) {
        bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0;
    } else {
        bbox[0] = results[0]; bbox[1] = results[1];
        bbox[2] = results[2] + 1; bbox[3] = results[3] + 1;
    }
    return GPU_OK;
}

/* ================================================================== */
/* getextrema: min/max per band (GPU parallel reduction)               */
/* ================================================================== */

static int
_ocl_getextrema(GPUBackend self, ImagingGPU im, void *extrema) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_getextrema_kernel) return GPU_ERROR_UNSUPPORTED;

    int bands = im->bands;
    /* Initialize: min=255, max=0 for each band */
    int init_vals[8]; /* max 4 bands * 2 */
    for (int b = 0; b < bands; b++) {
        init_vals[b * 2] = 255;     /* min starts at 255 */
        init_vals[b * 2 + 1] = 0;   /* max starts at 0 */
    }

    cl_int clerr;
    cl_mem result_buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        bands * 2 * sizeof(int), init_vals, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    int num_pixels = im->xsize * im->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    size_t local = 256;

    clSetKernelArg(ctx->k_getextrema_kernel, 0, sizeof(cl_mem), &im->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_getextrema_kernel, 1, sizeof(cl_mem), &result_buf);
    clSetKernelArg(ctx->k_getextrema_kernel, 2, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_getextrema_kernel, 3, sizeof(int), &im->pixelsize);
    clSetKernelArg(ctx->k_getextrema_kernel, 4, sizeof(int), &bands);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_getextrema_kernel,
                                    1, NULL, &global, &local, 0, NULL, NULL);
    if (clerr != CL_SUCCESS) { clReleaseMemObject(result_buf); return GPU_ERROR_LAUNCH; }

    int results[8];
    clEnqueueReadBuffer(ctx->queue, result_buf, CL_TRUE, 0,
                        bands * 2 * sizeof(int), results, 0, NULL, NULL);
    clReleaseMemObject(result_buf);

    UINT8 *ext = (UINT8 *)extrema;
    for (int b = 0; b < bands; b++) {
        ext[b * 2] = (UINT8)results[b * 2];
        ext[b * 2 + 1] = (UINT8)results[b * 2 + 1];
    }
    return GPU_OK;
}

/* ================================================================== */
/* effect_spread: random pixel displacement (GPU parallel)             */
/* ================================================================== */

static int
_ocl_effect_spread(GPUBackend self, ImagingGPU out, ImagingGPU in, int distance) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_effect_spread_kernel) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    size_t local = 256;

    /* Use a random seed; deterministic per call but varied across calls */
    unsigned int seed = (unsigned int)(num_pixels * 7919 + 12345);

    clSetKernelArg(ctx->k_effect_spread_kernel, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_effect_spread_kernel, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_effect_spread_kernel, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_effect_spread_kernel, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_effect_spread_kernel, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_effect_spread_kernel, 5, sizeof(int), &distance);
    clSetKernelArg(ctx->k_effect_spread_kernel, 6, sizeof(unsigned int), &seed);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_effect_spread_kernel,
                                           1, NULL, &global, &local, 0, NULL, NULL);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* NEW KERNEL SOURCES                                                   */
/* ================================================================== */

static const char *OPENCL_KERNEL_COLOR_MATRIX =
"__kernel void color_matrix(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int num_pixels, int pixelsize,\n"
"    float m00, float m01, float m02, float m03,\n"
"    float m10, float m11, float m12, float m13,\n"
"    float m20, float m21, float m22, float m23,\n"
"    float m30, float m31, float m32, float m33)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * pixelsize;\n"
"    float r = (float)input[base];\n"
"    float g = (pixelsize >= 2) ? (float)input[base+1] : 0.0f;\n"
"    float b = (pixelsize >= 3) ? (float)input[base+2] : 0.0f;\n"
"    float a = (pixelsize >= 4) ? (float)input[base+3] : 255.0f;\n"
"    float nr = m00*r + m01*g + m02*b + m03*a;\n"
"    float ng = m10*r + m11*g + m12*b + m13*a;\n"
"    float nb = m20*r + m21*g + m22*b + m23*a;\n"
"    float na = m30*r + m31*g + m32*b + m33*a;\n"
"    output[base] = (uchar)clamp((int)(nr + 0.5f), 0, 255);\n"
"    if (pixelsize >= 2) output[base+1] = (uchar)clamp((int)(ng + 0.5f), 0, 255);\n"
"    if (pixelsize >= 3) output[base+2] = (uchar)clamp((int)(nb + 0.5f), 0, 255);\n"
"    if (pixelsize >= 4) output[base+3] = (uchar)clamp((int)(na + 0.5f), 0, 255);\n"
"}\n";

static const char *OPENCL_KERNEL_COLOR_LUT3D =
"__kernel void color_lut_3d(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const short *table,\n"
"    int num_pixels, int pixelsize,\n"
"    int table_channels, int size1, int size2, int size3)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * pixelsize;\n"
"    float r = (float)input[base] / 255.0f * (float)(size1 - 1);\n"
"    float g = (float)input[base+1] / 255.0f * (float)(size2 - 1);\n"
"    float b = (float)input[base+2] / 255.0f * (float)(size3 - 1);\n"
"    int r0 = (int)r; int g0 = (int)g; int b0 = (int)b;\n"
"    int r1 = min(r0+1, size1-1); int g1 = min(g0+1, size2-1); int b1 = min(b0+1, size3-1);\n"
"    float fr = r - (float)r0; float fg = g - (float)g0; float fb = b - (float)b0;\n"
"    for (int c = 0; c < table_channels && c < pixelsize; c++) {\n"
"        float c000 = (float)table[((r0*size2+g0)*size3+b0)*table_channels+c];\n"
"        float c001 = (float)table[((r0*size2+g0)*size3+b1)*table_channels+c];\n"
"        float c010 = (float)table[((r0*size2+g1)*size3+b0)*table_channels+c];\n"
"        float c011 = (float)table[((r0*size2+g1)*size3+b1)*table_channels+c];\n"
"        float c100 = (float)table[((r1*size2+g0)*size3+b0)*table_channels+c];\n"
"        float c101 = (float)table[((r1*size2+g0)*size3+b1)*table_channels+c];\n"
"        float c110 = (float)table[((r1*size2+g1)*size3+b0)*table_channels+c];\n"
"        float c111 = (float)table[((r1*size2+g1)*size3+b1)*table_channels+c];\n"
"        float val = c000*(1-fr)*(1-fg)*(1-fb) + c001*(1-fr)*(1-fg)*fb\n"
"                  + c010*(1-fr)*fg*(1-fb)     + c011*(1-fr)*fg*fb\n"
"                  + c100*fr*(1-fg)*(1-fb)     + c101*fr*(1-fg)*fb\n"
"                  + c110*fr*fg*(1-fb)         + c111*fr*fg*fb;\n"
"        output[base+c] = (uchar)clamp((int)(val/255.0f + 0.5f), 0, 255);\n"
"    }\n"
"    if (pixelsize == 4 && table_channels < 4)\n"
"        output[base+3] = input[base+3];\n"
"}\n";

static const char *OPENCL_KERNEL_CROP =
"__kernel void crop_region(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int out_w, int out_h,\n"
"    int pixelsize, int x0, int y0)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int dy = gid / out_w;\n"
"    int dx = gid % out_w;\n"
"    if (dy >= out_h) return;\n"
"    int src_off = (y0 + dy) * in_w * pixelsize + (x0 + dx) * pixelsize;\n"
"    int dst_off = dy * out_w * pixelsize + dx * pixelsize;\n"
"    for (int c = 0; c < pixelsize; c++)\n"
"        output[dst_off + c] = input[src_off + c];\n"
"}\n";

static const char *OPENCL_KERNEL_EXPAND =
"__kernel void expand_image(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h,\n"
"    int out_w, int out_h,\n"
"    int pixelsize, int pad_x, int pad_y,\n"
"    int fill0, int fill1, int fill2, int fill3)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int dy = gid / out_w;\n"
"    int dx = gid % out_w;\n"
"    if (dy >= out_h) return;\n"
"    int dst_off = dy * out_w * pixelsize + dx * pixelsize;\n"
"    int sx = dx - pad_x, sy = dy - pad_y;\n"
"    if (sx >= 0 && sx < in_w && sy >= 0 && sy < in_h) {\n"
"        int src_off = sy * in_w * pixelsize + sx * pixelsize;\n"
"        for (int c = 0; c < pixelsize; c++)\n"
"            output[dst_off + c] = input[src_off + c];\n"
"    } else {\n"
"        if (pixelsize >= 1) output[dst_off] = (uchar)fill0;\n"
"        if (pixelsize >= 2) output[dst_off+1] = (uchar)fill1;\n"
"        if (pixelsize >= 3) output[dst_off+2] = (uchar)fill2;\n"
"        if (pixelsize >= 4) output[dst_off+3] = (uchar)fill3;\n"
"    }\n"
"}\n";

static const char *OPENCL_KERNEL_OFFSET =
"__kernel void offset_image(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int pixelsize,\n"
"    int xoffset, int yoffset)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (y >= height) return;\n"
"    int sx = (x - xoffset + width) % width;\n"
"    int sy = (y - yoffset + height) % height;\n"
"    int dst_off = y * width * pixelsize + x * pixelsize;\n"
"    int src_off = sy * width * pixelsize + sx * pixelsize;\n"
"    for (int c = 0; c < pixelsize; c++)\n"
"        output[dst_off + c] = input[src_off + c];\n"
"}\n";

static const char *OPENCL_KERNEL_GRADIENT =
"__kernel void linear_gradient(\n"
"    __global uchar *output,\n"
"    int width, int height, int direction)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (y >= height) return;\n"
"    uchar val = (direction == 0)\n"
"        ? (uchar)(y * 255 / (height > 1 ? height - 1 : 1))\n"
"        : (uchar)(x * 255 / (width > 1 ? width - 1 : 1));\n"
"    output[gid] = val;\n"
"}\n"
"\n"
"__kernel void radial_gradient(\n"
"    __global uchar *output,\n"
"    int width, int height)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    if (y >= height) return;\n"
"    float cx = (float)width * 0.5f;\n"
"    float cy = (float)height * 0.5f;\n"
"    float fdx = (float)x - cx;\n"
"    float fdy = (float)y - cy;\n"
"    float maxr = sqrt(cx * cx + cy * cy);\n"
"    float r = sqrt(fdx * fdx + fdy * fdy);\n"
"    uchar val = (uchar)clamp((int)(r / maxr * 255.0f + 0.5f), 0, 255);\n"
"    output[gid] = val;\n"
"}\n";

static const char *OPENCL_KERNEL_NEGATIVE =
"__kernel void negative(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int total_bytes)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    output[i] = 255 - input[i];\n"
"}\n";

static const char *OPENCL_KERNEL_POSTERIZE_SOLARIZE =
"__kernel void posterize(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int total_bytes, int bits)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    uchar mask = (uchar)(0xFF << (8 - bits));\n"
"    output[i] = input[i] & mask;\n"
"}\n"
"\n"
"__kernel void solarize(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int total_bytes, int threshold)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= total_bytes) return;\n"
"    uchar v = input[i];\n"
"    output[i] = (v >= threshold) ? (255 - v) : v;\n"
"}\n";

static const char *OPENCL_KERNEL_EQUALIZE =
"__kernel void equalize(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    __global const uchar *lut,\n"
"    int num_pixels, int pixelsize, int bands)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * pixelsize;\n"
"    for (int b = 0; b < bands; b++)\n"
"        output[base + b] = lut[b * 256 + input[base + b]];\n"
"    if (pixelsize > bands)\n"
"        output[base + pixelsize - 1] = input[base + pixelsize - 1];\n"
"}\n";

/* ================================================================== */
/* GPU-native getbbox / getextrema / effect_spread / convert_ycbcr /   */
/* reduce / rank_filter / mode_filter kernels                          */
/* ================================================================== */

static const char *OPENCL_KERNEL_GETBBOX =
"/* Parallel bounding box reduction: each work-item processes one pixel,\n"
"   atomically updates global min/max coordinates */\n"
"__kernel void getbbox(\n"
"    __global const uchar *input,\n"
"    __global volatile int *result,  /* [min_x, min_y, max_x, max_y] */\n"
"    int width, int height, int pixelsize, int check_channel)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    int base = gid * pixelsize;\n"
"    int nonzero = 0;\n"
"    if (check_channel >= 0) {\n"
"        nonzero = (input[base + check_channel] != 0);\n"
"    } else {\n"
"        for (int c = 0; c < pixelsize; c++)\n"
"            if (input[base + c] != 0) { nonzero = 1; break; }\n"
"    }\n"
"    if (nonzero) {\n"
"        atomic_min(&result[0], x);\n"
"        atomic_min(&result[1], y);\n"
"        atomic_max(&result[2], x);\n"
"        atomic_max(&result[3], y);\n"
"    }\n"
"}\n";

static const char *OPENCL_KERNEL_GETEXTREMA =
"/* Parallel min/max per band reduction */\n"
"__kernel void getextrema(\n"
"    __global const uchar *input,\n"
"    __global volatile int *result,  /* [min0,max0,min1,max1,...] */\n"
"    int num_pixels, int pixelsize, int bands)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int base = gid * pixelsize;\n"
"    for (int b = 0; b < bands; b++) {\n"
"        int v = (int)input[base + b];\n"
"        atomic_min(&result[b * 2], v);\n"
"        atomic_max(&result[b * 2 + 1], v);\n"
"    }\n"
"}\n";

static const char *OPENCL_KERNEL_EFFECT_SPREAD =
"/* Parallel effect_spread with hash-based PRNG (deterministic per pixel) */\n"
"__kernel void effect_spread(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int pixelsize, int distance, uint seed)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    /* Hash-based PRNG: Robert Jenkins' mix */\n"
"    uint h = (uint)gid ^ seed;\n"
"    h = (h + 0x7ed55d16u) + (h << 12);\n"
"    h = (h ^ 0xc761c23cu) ^ (h >> 19);\n"
"    h = (h + 0x165667b1u) + (h << 5);\n"
"    h = (h + 0xd3a2646cu) ^ (h << 9);\n"
"    h = (h + 0xfd7046c5u) + (h << 3);\n"
"    h = (h ^ 0xb55a4f09u) ^ (h >> 16);\n"
"    int diam = 2 * distance + 1;\n"
"    int dx = (int)(h % (uint)diam) - distance;\n"
"    h = h * 2654435761u + 0x12345u;\n"
"    int dy = (int)(h % (uint)diam) - distance;\n"
"    int sx = clamp(x + dx, 0, width - 1);\n"
"    int sy = clamp(y + dy, 0, height - 1);\n"
"    int src_off = (sy * width + sx) * pixelsize;\n"
"    int dst_off = gid * pixelsize;\n"
"    for (int c = 0; c < pixelsize; c++)\n"
"        output[dst_off + c] = input[src_off + c];\n"
"}\n";

static const char *OPENCL_KERNEL_CONVERT_YCBCR =
"/* YCbCr -> RGB conversion (ITU-R BT.601) */\n"
"__kernel void convert_ycbcr_to_rgb(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int num_pixels, int in_ps, int out_ps)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int in_base = gid * in_ps;\n"
"    int out_base = gid * out_ps;\n"
"    float y_val  = (float)input[in_base];\n"
"    float cb_val = (float)input[in_base + 1] - 128.0f;\n"
"    float cr_val = (float)input[in_base + 2] - 128.0f;\n"
"    int r = (int)(y_val + 1.402f * cr_val + 0.5f);\n"
"    int g = (int)(y_val - 0.344136f * cb_val - 0.714136f * cr_val + 0.5f);\n"
"    int b = (int)(y_val + 1.772f * cb_val + 0.5f);\n"
"    output[out_base]     = (uchar)clamp(r, 0, 255);\n"
"    output[out_base + 1] = (uchar)clamp(g, 0, 255);\n"
"    output[out_base + 2] = (uchar)clamp(b, 0, 255);\n"
"    if (out_ps > 3) output[out_base + 3] = 255;\n"
"}\n"
"\n"
"/* RGB -> YCbCr conversion (ITU-R BT.601) */\n"
"__kernel void convert_rgb_to_ycbcr(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int num_pixels, int in_ps, int out_ps)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_pixels) return;\n"
"    int in_base = gid * in_ps;\n"
"    int out_base = gid * out_ps;\n"
"    float r = (float)input[in_base];\n"
"    float g = (float)input[in_base + 1];\n"
"    float b = (float)input[in_base + 2];\n"
"    int y_val  = (int)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);\n"
"    int cb_val = (int)(-0.168736f * r - 0.331264f * g + 0.5f * b + 128.5f);\n"
"    int cr_val = (int)(0.5f * r - 0.418688f * g - 0.081312f * b + 128.5f);\n"
"    output[out_base]     = (uchar)clamp(y_val, 0, 255);\n"
"    output[out_base + 1] = (uchar)clamp(cb_val, 0, 255);\n"
"    output[out_base + 2] = (uchar)clamp(cr_val, 0, 255);\n"
"    if (out_ps > 3) output[out_base + 3] = 255;\n"
"}\n";

static const char *OPENCL_KERNEL_REDUCE =
"/* Integer-ratio downscale by averaging NxN blocks */\n"
"__kernel void reduce(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int in_w, int in_h, int out_w, int out_h,\n"
"    int factor_x, int factor_y, int pixelsize, int bands)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= out_w * out_h) return;\n"
"    int oy = gid / out_w;\n"
"    int ox = gid % out_w;\n"
"    int in_linesize = in_w * pixelsize;\n"
"    int out_linesize = out_w * pixelsize;\n"
"    for (int b = 0; b < bands; b++) {\n"
"        int acc = 0;\n"
"        int count = 0;\n"
"        for (int dy = 0; dy < factor_y; dy++) {\n"
"            int iy = oy * factor_y + dy;\n"
"            if (iy >= in_h) break;\n"
"            for (int dx = 0; dx < factor_x; dx++) {\n"
"                int ix = ox * factor_x + dx;\n"
"                if (ix >= in_w) break;\n"
"                acc += (int)input[iy * in_linesize + ix * pixelsize + b];\n"
"                count++;\n"
"            }\n"
"        }\n"
"        output[oy * out_linesize + ox * pixelsize + b] =\n"
"            (uchar)((acc + count / 2) / count);\n"
"    }\n"
"    /* Copy alpha if pixelsize > bands (e.g. RGBX) */\n"
"    if (pixelsize > bands) {\n"
"        int iy = oy * factor_y;\n"
"        int ix = ox * factor_x;\n"
"        if (iy < in_h && ix < in_w)\n"
"            output[oy * out_linesize + ox * pixelsize + pixelsize - 1] =\n"
"                input[iy * in_linesize + ix * pixelsize + pixelsize - 1];\n"
"    }\n"
"}\n";

static const char *OPENCL_KERNEL_RANK_FILTER =
"/* Rank filter: min(rank=0), median, max within NxN window */\n"
"__kernel void rank_filter(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int pixelsize, int bands,\n"
"    int ksize, int rank)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    int khalf = ksize / 2;\n"
"    int linesize = width * pixelsize;\n"
"    for (int b = 0; b < bands; b++) {\n"
"        /* Collect values in window */\n"
"        uchar vals[49]; /* max 7x7 window */\n"
"        int count = 0;\n"
"        for (int dy = -khalf; dy <= khalf; dy++) {\n"
"            int iy = clamp(y + dy, 0, height - 1);\n"
"            for (int dx = -khalf; dx <= khalf; dx++) {\n"
"                int ix = clamp(x + dx, 0, width - 1);\n"
"                vals[count++] = input[iy * linesize + ix * pixelsize + b];\n"
"            }\n"
"        }\n"
"        /* Partial insertion sort to find rank-th element */\n"
"        for (int i = 0; i <= rank && i < count; i++) {\n"
"            int min_idx = i;\n"
"            for (int j = i + 1; j < count; j++) {\n"
"                if (vals[j] < vals[min_idx]) min_idx = j;\n"
"            }\n"
"            uchar tmp = vals[i]; vals[i] = vals[min_idx]; vals[min_idx] = tmp;\n"
"        }\n"
"        output[y * linesize + x * pixelsize + b] = vals[rank];\n"
"    }\n"
"    if (pixelsize > bands)\n"
"        output[y * linesize + x * pixelsize + pixelsize - 1] =\n"
"            input[y * linesize + x * pixelsize + pixelsize - 1];\n"
"}\n";

static const char *OPENCL_KERNEL_MODE_FILTER =
"/* Mode filter: most frequent value in NxN window (L-mode) */\n"
"__kernel void mode_filter(\n"
"    __global const uchar *input,\n"
"    __global uchar *output,\n"
"    int width, int height, int ksize)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= width * height) return;\n"
"    int y = gid / width;\n"
"    int x = gid % width;\n"
"    int khalf = ksize / 2;\n"
"    /* Histogram of window values */\n"
"    uchar hist[256];\n"
"    for (int i = 0; i < 256; i++) hist[i] = 0;\n"
"    for (int dy = -khalf; dy <= khalf; dy++) {\n"
"        int iy = clamp(y + dy, 0, height - 1);\n"
"        for (int dx = -khalf; dx <= khalf; dx++) {\n"
"            int ix = clamp(x + dx, 0, width - 1);\n"
"            hist[input[iy * width + ix]]++;\n"
"        }\n"
"    }\n"
"    /* Find mode */\n"
"    uchar best_val = input[y * width + x]; /* default: center pixel */\n"
"    uchar best_count = 1;\n"
"    for (int i = 0; i < 256; i++) {\n"
"        if (hist[i] > best_count) {\n"
"            best_count = hist[i];\n"
"            best_val = (uchar)i;\n"
"        }\n"
"    }\n"
"    output[y * width + x] = best_val;\n"
"}\n";

static int
_ocl_color_matrix(GPUBackend self, ImagingGPU out, ImagingGPU in,
                  const float *matrix, int ncolumns) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_color_matrix) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    /* Expand matrix to 4x4, fill missing with identity */
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    for (int r = 0; r < 4 && r < ncolumns; r++)
        for (int c = 0; c < 4 && c < ncolumns; c++)
            m[r * 4 + c] = matrix[r * ncolumns + c];

    clSetKernelArg(ctx->k_color_matrix, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_color_matrix, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_color_matrix, 2, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_color_matrix, 3, sizeof(int), &in->pixelsize);
    for (int i = 0; i < 16; i++)
        clSetKernelArg(ctx->k_color_matrix, 4 + i, sizeof(float), &m[i]);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_color_matrix, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* color_lut_3d implementation                                          */
/* ================================================================== */

static int
_ocl_color_lut_3d(GPUBackend self, ImagingGPU out, ImagingGPU in,
                  int table_channels, int size1D, int size2D, int size3D,
                  const INT16 *table) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_color_lut_3d) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    int table_size = size1D * size2D * size3D * table_channels;

    cl_int clerr;
    cl_mem table_buf = clCreateBuffer(ctx->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       table_size * sizeof(INT16),
                                       (void *)table, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    clSetKernelArg(ctx->k_color_lut_3d, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_color_lut_3d, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_color_lut_3d, 2, sizeof(cl_mem), &table_buf);
    clSetKernelArg(ctx->k_color_lut_3d, 3, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_color_lut_3d, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_color_lut_3d, 5, sizeof(int), &table_channels);
    clSetKernelArg(ctx->k_color_lut_3d, 6, sizeof(int), &size1D);
    clSetKernelArg(ctx->k_color_lut_3d, 7, sizeof(int), &size2D);
    clSetKernelArg(ctx->k_color_lut_3d, 8, sizeof(int), &size3D);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_color_lut_3d, 1,
                                    NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    clReleaseMemObject(table_buf);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* crop implementation                                                  */
/* ================================================================== */

static int
_ocl_crop(GPUBackend self, ImagingGPU out, ImagingGPU in,
          int x0, int y0, int x1, int y1) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_crop_region) return GPU_ERROR_UNSUPPORTED;

    int out_w = x1 - x0;
    int out_h = y1 - y0;
    int num_pixels = out_w * out_h;
    if (num_pixels <= 0) return GPU_OK;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_crop_region, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_crop_region, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_crop_region, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_crop_region, 3, sizeof(int), &out_w);
    clSetKernelArg(ctx->k_crop_region, 4, sizeof(int), &out_h);
    clSetKernelArg(ctx->k_crop_region, 5, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_crop_region, 6, sizeof(int), &x0);
    clSetKernelArg(ctx->k_crop_region, 7, sizeof(int), &y0);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_crop_region, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* expand implementation                                                */
/* ================================================================== */

static int
_ocl_expand(GPUBackend self, ImagingGPU out, ImagingGPU in,
            int xmargin, int ymargin, const UINT8 *fill) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_expand_image) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    int f0 = fill ? fill[0] : 0;
    int f1 = fill ? (in->pixelsize >= 2 ? fill[1] : 0) : 0;
    int f2 = fill ? (in->pixelsize >= 3 ? fill[2] : 0) : 0;
    int f3 = fill ? (in->pixelsize >= 4 ? fill[3] : 0) : 0;

    clSetKernelArg(ctx->k_expand_image, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_expand_image, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_expand_image, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_expand_image, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_expand_image, 4, sizeof(int), &out->xsize);
    clSetKernelArg(ctx->k_expand_image, 5, sizeof(int), &out->ysize);
    clSetKernelArg(ctx->k_expand_image, 6, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_expand_image, 7, sizeof(int), &xmargin);
    clSetKernelArg(ctx->k_expand_image, 8, sizeof(int), &ymargin);
    clSetKernelArg(ctx->k_expand_image, 9, sizeof(int), &f0);
    clSetKernelArg(ctx->k_expand_image, 10, sizeof(int), &f1);
    clSetKernelArg(ctx->k_expand_image, 11, sizeof(int), &f2);
    clSetKernelArg(ctx->k_expand_image, 12, sizeof(int), &f3);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_expand_image, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* offset implementation                                                */
/* ================================================================== */

static int
_ocl_offset(GPUBackend self, ImagingGPU out, ImagingGPU in,
            int xoffset, int yoffset) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_offset_image) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_offset_image, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_offset_image, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_offset_image, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_offset_image, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_offset_image, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_offset_image, 5, sizeof(int), &xoffset);
    clSetKernelArg(ctx->k_offset_image, 6, sizeof(int), &yoffset);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_offset_image, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* negative implementation                                              */
/* ================================================================== */

static int
_ocl_negative(GPUBackend self, ImagingGPU out, ImagingGPU in) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_negative) return GPU_ERROR_UNSUPPORTED;

    int total_bytes = in->xsize * in->ysize * in->pixelsize;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_negative, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_negative, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_negative, 2, sizeof(int), &total_bytes);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_negative, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* posterize implementation                                             */
/* ================================================================== */

static int
_ocl_posterize(GPUBackend self, ImagingGPU out, ImagingGPU in, int bits) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_posterize) return GPU_ERROR_UNSUPPORTED;

    int total_bytes = in->xsize * in->ysize * in->pixelsize;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_posterize, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_posterize, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_posterize, 2, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_posterize, 3, sizeof(int), &bits);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_posterize, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* solarize implementation                                              */
/* ================================================================== */

static int
_ocl_solarize(GPUBackend self, ImagingGPU out, ImagingGPU in, int threshold) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_solarize) return GPU_ERROR_UNSUPPORTED;

    int total_bytes = in->xsize * in->ysize * in->pixelsize;
    size_t global = _ocl_round_up(total_bytes, 256);

    clSetKernelArg(ctx->k_solarize, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_solarize, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_solarize, 2, sizeof(int), &total_bytes);
    clSetKernelArg(ctx->k_solarize, 3, sizeof(int), &threshold);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_solarize, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* equalize implementation                                              */
/* ================================================================== */

static int
_ocl_equalize(GPUBackend self, ImagingGPU out, ImagingGPU in,
              const UINT8 *lut) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_equalize) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    int bands = in->bands;
    int lut_size = bands * 256;
    size_t global = _ocl_round_up(num_pixels, 256);

    cl_int clerr;
    cl_mem lut_buf = clCreateBuffer(ctx->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     lut_size, (void *)lut, &clerr);
    if (clerr != CL_SUCCESS) return GPU_ERROR_MEMORY;

    clSetKernelArg(ctx->k_equalize, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_equalize, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_equalize, 2, sizeof(cl_mem), &lut_buf);
    clSetKernelArg(ctx->k_equalize, 3, sizeof(int), &num_pixels);
    clSetKernelArg(ctx->k_equalize, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_equalize, 5, sizeof(int), &bands);

    clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_equalize, 1,
                                    NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    clReleaseMemObject(lut_buf);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* gradient implementations                                             */
/* ================================================================== */

static int
_ocl_linear_gradient(GPUBackend self, ImagingGPU out, int direction) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_linear_gradient) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_linear_gradient, 0, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_linear_gradient, 1, sizeof(int), &out->xsize);
    clSetKernelArg(ctx->k_linear_gradient, 2, sizeof(int), &out->ysize);
    clSetKernelArg(ctx->k_linear_gradient, 3, sizeof(int), &direction);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_linear_gradient, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

static int
_ocl_radial_gradient(GPUBackend self, ImagingGPU out) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_radial_gradient) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);

    clSetKernelArg(ctx->k_radial_gradient, 0, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_radial_gradient, 1, sizeof(int), &out->xsize);
    clSetKernelArg(ctx->k_radial_gradient, 2, sizeof(int), &out->ysize);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_radial_gradient, 1,
                                           NULL, &global, NULL, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* reduce: integer downscale (GPU parallel per output pixel)           */
/* ================================================================== */

static int
_ocl_reduce(GPUBackend self, ImagingGPU out, ImagingGPU in,
            int factor_x, int factor_y) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_reduce) return GPU_ERROR_UNSUPPORTED;

    int out_pixels = out->xsize * out->ysize;
    size_t global = _ocl_round_up(out_pixels, 256);
    size_t local = 256;
    int bands = in->bands;

    clSetKernelArg(ctx->k_reduce, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_reduce, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_reduce, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_reduce, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_reduce, 4, sizeof(int), &out->xsize);
    clSetKernelArg(ctx->k_reduce, 5, sizeof(int), &out->ysize);
    clSetKernelArg(ctx->k_reduce, 6, sizeof(int), &factor_x);
    clSetKernelArg(ctx->k_reduce, 7, sizeof(int), &factor_y);
    clSetKernelArg(ctx->k_reduce, 8, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_reduce, 9, sizeof(int), &bands);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_reduce,
                                           1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* rank_filter: min/median/max in NxN window (GPU parallel per pixel)  */
/* ================================================================== */

static int
_ocl_rank_filter(GPUBackend self, ImagingGPU out, ImagingGPU in,
                 int ksize, int rank) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_rank_filter) return GPU_ERROR_UNSUPPORTED;
    if (ksize > 7) return GPU_ERROR_UNSUPPORTED; /* kernel supports max 7x7 */

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    size_t local = 256;
    int bands = in->bands;

    clSetKernelArg(ctx->k_rank_filter, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_rank_filter, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_rank_filter, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_rank_filter, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_rank_filter, 4, sizeof(int), &in->pixelsize);
    clSetKernelArg(ctx->k_rank_filter, 5, sizeof(int), &bands);
    clSetKernelArg(ctx->k_rank_filter, 6, sizeof(int), &ksize);
    clSetKernelArg(ctx->k_rank_filter, 7, sizeof(int), &rank);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_rank_filter,
                                           1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* mode_filter: most frequent value in NxN window (GPU parallel, L)    */
/* ================================================================== */

static int
_ocl_mode_filter(GPUBackend self, ImagingGPU out, ImagingGPU in, int ksize) {
    OpenCLContext *ctx = (OpenCLContext *)self->ctx;
    if (!ctx->k_mode_filter) return GPU_ERROR_UNSUPPORTED;
    /* mode_filter only supports L (1-band) images */
    if (in->bands != 1) return GPU_ERROR_UNSUPPORTED;

    int num_pixels = in->xsize * in->ysize;
    size_t global = _ocl_round_up(num_pixels, 256);
    size_t local = 256;

    clSetKernelArg(ctx->k_mode_filter, 0, sizeof(cl_mem), &in->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_mode_filter, 1, sizeof(cl_mem), &out->buffer.handle.cl_mem);
    clSetKernelArg(ctx->k_mode_filter, 2, sizeof(int), &in->xsize);
    clSetKernelArg(ctx->k_mode_filter, 3, sizeof(int), &in->ysize);
    clSetKernelArg(ctx->k_mode_filter, 4, sizeof(int), &ksize);

    cl_int clerr = clEnqueueNDRangeKernel(ctx->queue, ctx->k_mode_filter,
                                           1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(ctx->queue);
    return (clerr == CL_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* ================================================================== */
/* Backend initialization                                               */
/* ================================================================== */

GPUBackend
ImagingGPU_OpenCL_Init(void) {
    cl_int err;

    /* Get platform */
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return NULL;

    /* Get GPU device (prefer discrete, fall back to any GPU) */
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
        if (err != CL_SUCCESS) return NULL;
    }

    /* Create context and command queue */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return NULL;

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        return NULL;
    }

    /* Concatenate all kernel sources */
    const char *sources[] = {
        OPENCL_KERNEL_COMMON,
        OPENCL_KERNEL_BLUR,
        OPENCL_KERNEL_FILTER,
        OPENCL_KERNEL_RESAMPLE,
        OPENCL_KERNEL_CONVERT,
        OPENCL_KERNEL_BLEND,
        OPENCL_KERNEL_CHOPS,
        OPENCL_KERNEL_GEOMETRY,
        OPENCL_KERNEL_POINT,
        OPENCL_KERNEL_BANDS,
        OPENCL_KERNEL_UNSHARP,
        OPENCL_KERNEL_PASTE,
        OPENCL_KERNEL_HISTOGRAM,
        OPENCL_KERNEL_TRANSFORM,
        OPENCL_KERNEL_COLOR_MATRIX,
        OPENCL_KERNEL_COLOR_LUT3D,
        OPENCL_KERNEL_CROP,
        OPENCL_KERNEL_EXPAND,
        OPENCL_KERNEL_OFFSET,
        OPENCL_KERNEL_GRADIENT,
        OPENCL_KERNEL_NEGATIVE,
        OPENCL_KERNEL_POSTERIZE_SOLARIZE,
        OPENCL_KERNEL_EQUALIZE,
        OPENCL_KERNEL_GETBBOX,
        OPENCL_KERNEL_GETEXTREMA,
        OPENCL_KERNEL_EFFECT_SPREAD,
        OPENCL_KERNEL_CONVERT_YCBCR,
        OPENCL_KERNEL_REDUCE,
        OPENCL_KERNEL_RANK_FILTER,
        OPENCL_KERNEL_MODE_FILTER,
    };
    int nsources = sizeof(sources) / sizeof(sources[0]);

    cl_program program = clCreateProgramWithSource(context, nsources, sources,
                                                    NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return NULL;
    }

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Print build log for debugging */
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            fprintf(stderr, "Pillow GPU OpenCL build log:\n%s\n", log);
            free(log);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return NULL;
    }

    /* Allocate context */
    OpenCLContext *ctx = (OpenCLContext *)calloc(1, sizeof(OpenCLContext));
    if (!ctx) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return NULL;
    }
    ctx->context = context;
    ctx->queue = queue;
    ctx->device = device;
    ctx->program = program;

    /* Create all kernels */
    ctx->k_box_blur_h = _ocl_create_kernel(program, "box_blur_h");
    ctx->k_box_blur_v = _ocl_create_kernel(program, "box_blur_v");
    ctx->k_gaussian_blur_h = _ocl_create_kernel(program, "gaussian_blur_h");
    ctx->k_gaussian_blur_v = _ocl_create_kernel(program, "gaussian_blur_v");
    ctx->k_convolve = _ocl_create_kernel(program, "convolve");
    ctx->k_resample_nearest = _ocl_create_kernel(program, "resample_nearest");
    ctx->k_resample_bilinear = _ocl_create_kernel(program, "resample_bilinear");
    ctx->k_resample_bicubic = _ocl_create_kernel(program, "resample_bicubic");
    ctx->k_resample_lanczos = _ocl_create_kernel(program, "resample_lanczos");
    ctx->k_convert_rgb_to_l = _ocl_create_kernel(program, "convert_rgb_to_l");
    ctx->k_convert_l_to_rgb = _ocl_create_kernel(program, "convert_l_to_rgb");
    ctx->k_convert_rgb_to_rgba = _ocl_create_kernel(program, "convert_rgb_to_rgba");
    ctx->k_convert_rgba_to_rgb = _ocl_create_kernel(program, "convert_rgba_to_rgb");
    ctx->k_convert_rgba_to_l = _ocl_create_kernel(program, "convert_rgba_to_l");
    ctx->k_convert_l_to_rgba = _ocl_create_kernel(program, "convert_l_to_rgba");
    ctx->k_convert_rgb_to_cmyk = _ocl_create_kernel(program, "convert_rgb_to_cmyk");
    ctx->k_convert_cmyk_to_rgb = _ocl_create_kernel(program, "convert_cmyk_to_rgb");
    ctx->k_blend = _ocl_create_kernel(program, "blend");
    ctx->k_alpha_composite = _ocl_create_kernel(program, "alpha_composite");
    ctx->k_chop_operation = _ocl_create_kernel(program, "chop_operation");
    ctx->k_transpose_op = _ocl_create_kernel(program, "transpose_op");
    ctx->k_point_lut = _ocl_create_kernel(program, "point_lut");
    ctx->k_point_transform = _ocl_create_kernel(program, "point_transform");
    ctx->k_getband = _ocl_create_kernel(program, "getband");
    ctx->k_putband = _ocl_create_kernel(program, "putband");
    ctx->k_fillband = _ocl_create_kernel(program, "fillband");
    ctx->k_fill_color = _ocl_create_kernel(program, "fill_color");
    ctx->k_merge_bands = _ocl_create_kernel(program, "merge_bands");
    ctx->k_unsharp_mask = _ocl_create_kernel(program, "unsharp_mask");
    ctx->k_paste_with_mask = _ocl_create_kernel(program, "paste_with_mask");
    ctx->k_histogram = _ocl_create_kernel(program, "histogram");
    ctx->k_affine_transform = _ocl_create_kernel(program, "affine_transform");
    ctx->k_perspective_transform = _ocl_create_kernel(program, "perspective_transform");
    ctx->k_color_matrix = _ocl_create_kernel(program, "color_matrix");
    ctx->k_color_lut_3d = _ocl_create_kernel(program, "color_lut_3d");
    ctx->k_crop_region = _ocl_create_kernel(program, "crop_region");
    ctx->k_expand_image = _ocl_create_kernel(program, "expand_image");
    ctx->k_offset_image = _ocl_create_kernel(program, "offset_image");
    ctx->k_linear_gradient = _ocl_create_kernel(program, "linear_gradient");
    ctx->k_radial_gradient = _ocl_create_kernel(program, "radial_gradient");
    ctx->k_negative = _ocl_create_kernel(program, "negative");
    ctx->k_posterize = _ocl_create_kernel(program, "posterize");
    ctx->k_solarize = _ocl_create_kernel(program, "solarize");
    ctx->k_equalize = _ocl_create_kernel(program, "equalize");
    ctx->k_getbbox_kernel = _ocl_create_kernel(program, "getbbox");
    ctx->k_getextrema_kernel = _ocl_create_kernel(program, "getextrema");
    ctx->k_effect_spread_kernel = _ocl_create_kernel(program, "effect_spread");
    ctx->k_convert_ycbcr_to_rgb = _ocl_create_kernel(program, "convert_ycbcr_to_rgb");
    ctx->k_convert_rgb_to_ycbcr = _ocl_create_kernel(program, "convert_rgb_to_ycbcr");
    ctx->k_reduce = _ocl_create_kernel(program, "reduce");
    ctx->k_rank_filter = _ocl_create_kernel(program, "rank_filter");
    ctx->k_mode_filter = _ocl_create_kernel(program, "mode_filter");

    /* Allocate backend struct */
    GPUBackend backend = (GPUBackend)calloc(1, sizeof(struct GPUBackendInstance));
    if (!backend) {
        _ocl_shutdown(NULL); /* will be reworked */
        free(ctx);
        return NULL;
    }

    backend->type = GPU_BACKEND_OPENCL;
    backend->name = "OpenCL";
    backend->ctx = ctx;

    /* Query device info */
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(backend->device_name),
                    backend->device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t),
                    &backend->max_mem_alloc, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t),
                    &backend->total_mem, NULL);

    /* Assign function pointers */
    backend->shutdown = _ocl_shutdown;
    backend->buffer_alloc = _ocl_buffer_alloc;
    backend->buffer_free = _ocl_buffer_free;
    backend->buffer_upload = _ocl_buffer_upload;
    backend->buffer_download = _ocl_buffer_download;
    backend->buffer_copy = _ocl_buffer_copy;
    backend->gaussian_blur = _ocl_gaussian_blur;
    backend->box_blur = _ocl_box_blur;
    backend->unsharp_mask = _ocl_unsharp_mask;
    backend->filter = _ocl_filter;
    backend->resample = _ocl_resample;
    backend->convert = _ocl_convert;
    backend->blend = _ocl_blend;
    backend->alpha_composite = _ocl_alpha_composite;
    backend->chop = _ocl_chop;
    backend->transpose = _ocl_transpose;
    backend->transform = _ocl_transform;
    backend->point_lut = _ocl_point_lut;
    backend->point_transform = _ocl_point_transform;
    backend->fill = _ocl_fill;
    backend->copy = _ocl_copy;
    backend->getband = _ocl_getband;
    backend->putband = _ocl_putband;
    backend->fillband = _ocl_fillband;
    backend->histogram = _ocl_histogram;

    /* Operations now fully implemented */
    backend->color_matrix = _ocl_color_matrix;
    backend->color_lut_3d = _ocl_color_lut_3d;
    backend->paste = _ocl_paste;
    backend->getbbox = _ocl_getbbox;
    backend->getextrema = _ocl_getextrema;
    backend->merge = _ocl_merge;
    backend->split = _ocl_split;
    backend->effect_spread = _ocl_effect_spread;

    /* Extended operations */
    backend->crop = _ocl_crop;
    backend->expand = _ocl_expand;
    backend->offset = _ocl_offset;
    backend->negative = _ocl_negative;
    backend->posterize = _ocl_posterize;
    backend->solarize = _ocl_solarize;
    backend->equalize = _ocl_equalize;
    backend->linear_gradient = _ocl_linear_gradient;
    backend->radial_gradient = _ocl_radial_gradient;
    backend->reduce = _ocl_reduce;
    backend->rank_filter = _ocl_rank_filter;
    backend->mode_filter = _ocl_mode_filter;

    return backend;
}

#endif /* HAVE_OPENCL */
