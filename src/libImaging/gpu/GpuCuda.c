/*
 * Pillow GPU Acceleration — CUDA Backend
 *
 * Implements GPU operations using CUDA Driver API + NVRTC.
 * Kernels compiled at runtime from embedded source strings.
 * No nvcc dependency at build time — only cuda.h, nvrtc.h, and
 * the CUDA runtime libraries are needed.
 *
 * Copyright (c) 2026 Pillow Contributors
 */

#ifdef HAVE_CUDA

#include <cuda.h>
#include <nvrtc.h>

#include "GpuImaging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ================================================================== */
/* CUDA kernel source (compiled with NVRTC at runtime)                  */
/* ================================================================== */

static const char *CUDA_KERNEL_SOURCE =
    "#include <cstdint>\n"
    "typedef unsigned char uchar;\n"
    "\n"
    "__device__ int gpu_clamp(int v, int lo, int hi) {\n"
    "    return v < lo ? lo : (v > hi ? hi : v);\n"
    "}\n"
    "\n"
    "#define CLIP8(v) gpu_clamp((int)(v), 0, 255)\n"
    "#define MULDIV255(a, b) (((a) * (b) + 128 + (((a) * (b) + 128) >> 8)) >> 8)\n"
    "\n"
    "/* ============ Box Blur ============ */\n"
    "\n"
    "extern \"C\" __global__ void box_blur_h(\n"
    "    const uchar *input, uchar *output,\n"
    "    int width, int height, int pixelsize, int radius)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = gid / width;\n"
    "    int x = gid % width;\n"
    "    if (x >= width || y >= height) return;\n"
    "    int linesize = width * pixelsize;\n"
    "    int diam = 2 * radius + 1;\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        int acc = 0;\n"
    "        for (int i = -radius; i <= radius; i++) {\n"
    "            int sx = gpu_clamp(x + i, 0, width - 1);\n"
    "            acc += input[y * linesize + sx * pixelsize + c];\n"
    "        }\n"
    "        output[y * linesize + x * pixelsize + c] =\n"
    "            (uchar)((acc + diam/2) / diam);\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void box_blur_v(\n"
    "    const uchar *input, uchar *output,\n"
    "    int width, int height, int pixelsize, int radius)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = gid / width;\n"
    "    int x = gid % width;\n"
    "    if (x >= width || y >= height) return;\n"
    "    int linesize = width * pixelsize;\n"
    "    int diam = 2 * radius + 1;\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        int acc = 0;\n"
    "        for (int i = -radius; i <= radius; i++) {\n"
    "            int sy = gpu_clamp(y + i, 0, height - 1);\n"
    "            acc += input[sy * linesize + x * pixelsize + c];\n"
    "        }\n"
    "        output[y * linesize + x * pixelsize + c] =\n"
    "            (uchar)((acc + diam/2) / diam);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Gaussian Blur ============ */\n"
    "\n"
    "extern \"C\" __global__ void gaussian_blur_h(\n"
    "    const uchar *input, uchar *output, const float *weights,\n"
    "    int width, int height, int pixelsize, int radius)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = gid / width;\n"
    "    int x = gid % width;\n"
    "    if (x >= width || y >= height) return;\n"
    "    int linesize = width * pixelsize;\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float sum = 0.0f;\n"
    "        for (int i = -radius; i <= radius; i++) {\n"
    "            int sx = gpu_clamp(x + i, 0, width - 1);\n"
    "            sum += (float)input[y * linesize + sx * pixelsize + c]\n"
    "                   * weights[i + radius];\n"
    "        }\n"
    "        output[y * linesize + x * pixelsize + c] =\n"
    "            (uchar)gpu_clamp((int)(sum + 0.5f), 0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void gaussian_blur_v(\n"
    "    const uchar *input, uchar *output, const float *weights,\n"
    "    int width, int height, int pixelsize, int radius)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = gid / width;\n"
    "    int x = gid % width;\n"
    "    if (x >= width || y >= height) return;\n"
    "    int linesize = width * pixelsize;\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float sum = 0.0f;\n"
    "        for (int i = -radius; i <= radius; i++) {\n"
    "            int sy = gpu_clamp(y + i, 0, height - 1);\n"
    "            sum += (float)input[sy * linesize + x * pixelsize + c]\n"
    "                   * weights[i + radius];\n"
    "        }\n"
    "        output[y * linesize + x * pixelsize + c] =\n"
    "            (uchar)gpu_clamp((int)(sum + 0.5f), 0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Convolution ============ */\n"
    "\n"
    "extern \"C\" __global__ void convolve(\n"
    "    const uchar *input, uchar *output, const float *kernel_data,\n"
    "    int width, int height, int pixelsize,\n"
    "    int kw, int kh, float divisor, float offset)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int y = gid / width;\n"
    "    int x = gid % width;\n"
    "    if (x >= width || y >= height) return;\n"
    "    int linesize = width * pixelsize;\n"
    "    int kw2 = kw / 2, kh2 = kh / 2;\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float sum = 0.0f;\n"
    "        for (int ky = 0; ky < kh; ky++) {\n"
    "            int sy = gpu_clamp(y + ky - kh2, 0, height - 1);\n"
    "            for (int kx = 0; kx < kw; kx++) {\n"
    "                int sx = gpu_clamp(x + kx - kw2, 0, width - 1);\n"
    "                sum += (float)input[sy * linesize + sx * pixelsize + c]\n"
    "                       * kernel_data[ky * kw + kx];\n"
    "            }\n"
    "        }\n"
    "        int val = (int)(sum / divisor + offset + 0.5f);\n"
    "        output[y * linesize + x * pixelsize + c] =\n"
    "            (uchar)gpu_clamp(val, 0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Resample ============ */\n"
    "\n"
    "extern \"C\" __global__ void resample_bilinear(\n"
    "    const uchar *input, uchar *output,\n"
    "    int in_w, int in_h, int out_w, int out_h, int pixelsize,\n"
    "    float box_x0, float box_y0, float box_x1, float box_y1)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int oy = gid / out_w;\n"
    "    int ox = gid % out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    int in_ls = in_w * pixelsize;\n"
    "    int out_ls = out_w * pixelsize;\n"
    "\n"
    "    float sx_f = box_x0 + (ox + 0.5f) * (box_x1 - box_x0) / out_w - 0.5f;\n"
    "    float sy_f = box_y0 + (oy + 0.5f) * (box_y1 - box_y0) / out_h - 0.5f;\n"
    "    int x0 = (int)floorf(sx_f), y0 = (int)floorf(sy_f);\n"
    "    float fx = sx_f - x0, fy = sy_f - y0;\n"
    "    int x1 = x0 + 1, y1 = y0 + 1;\n"
    "    x0 = gpu_clamp(x0, 0, in_w-1); x1 = gpu_clamp(x1, 0, in_w-1);\n"
    "    y0 = gpu_clamp(y0, 0, in_h-1); y1 = gpu_clamp(y1, 0, in_h-1);\n"
    "\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float v00 = input[y0*in_ls + x0*pixelsize + c];\n"
    "        float v10 = input[y0*in_ls + x1*pixelsize + c];\n"
    "        float v01 = input[y1*in_ls + x0*pixelsize + c];\n"
    "        float v11 = input[y1*in_ls + x1*pixelsize + c];\n"
    "        float val = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + "
    "v11*fx*fy;\n"
    "        output[oy*out_ls + ox*pixelsize + c] = (uchar)gpu_clamp((int)(val+0.5f), "
    "0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "__device__ float cubic_weight(float x) {\n"
    "    float ax = fabsf(x);\n"
    "    if (ax < 1.0f) return (1.5f*ax - 2.5f)*ax*ax + 1.0f;\n"
    "    if (ax < 2.0f) return ((-0.5f*ax + 2.5f)*ax - 4.0f)*ax + 2.0f;\n"
    "    return 0.0f;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void resample_bicubic(\n"
    "    const uchar *input, uchar *output,\n"
    "    int in_w, int in_h, int out_w, int out_h, int pixelsize,\n"
    "    float box_x0, float box_y0, float box_x1, float box_y1)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int oy = gid / out_w, ox = gid % out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    int in_ls = in_w*pixelsize, out_ls = out_w*pixelsize;\n"
    "    float sx = box_x0 + (ox+0.5f)*(box_x1-box_x0)/out_w - 0.5f;\n"
    "    float sy = box_y0 + (oy+0.5f)*(box_y1-box_y0)/out_h - 0.5f;\n"
    "    int ix = (int)floorf(sx), iy = (int)floorf(sy);\n"
    "    float fx = sx-ix, fy = sy-iy;\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float sum = 0.0f;\n"
    "        for (int j = -1; j <= 2; j++) {\n"
    "            float wy = cubic_weight(fy-j);\n"
    "            int csy = gpu_clamp(iy+j, 0, in_h-1);\n"
    "            for (int i = -1; i <= 2; i++) {\n"
    "                float wx = cubic_weight(fx-i);\n"
    "                int csx = gpu_clamp(ix+i, 0, in_w-1);\n"
    "                sum += input[csy*in_ls + csx*pixelsize + c] * wx * wy;\n"
    "            }\n"
    "        }\n"
    "        output[oy*out_ls + ox*pixelsize + c] = (uchar)gpu_clamp((int)(sum+0.5f), "
    "0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "__device__ float gpu_sinc(float x) {\n"
    "    if (fabsf(x) < 1e-6f) return 1.0f;\n"
    "    float px = 3.14159265358979f * x;\n"
    "    return sinf(px) / px;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void resample_lanczos(\n"
    "    const uchar *input, uchar *output,\n"
    "    int in_w, int in_h, int out_w, int out_h, int pixelsize,\n"
    "    float box_x0, float box_y0, float box_x1, float box_y1)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int oy = gid / out_w, ox = gid % out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    int in_ls = in_w*pixelsize, out_ls = out_w*pixelsize;\n"
    "    float scx = (box_x1-box_x0)/out_w, scy = (box_y1-box_y0)/out_h;\n"
    "    float sx = box_x0 + (ox+0.5f)*scx - 0.5f;\n"
    "    float sy = box_y0 + (oy+0.5f)*scy - 0.5f;\n"
    "    int a = 3;\n"
    "    float supx = fmaxf((float)a, (float)a*scx);\n"
    "    float supy = fmaxf((float)a, (float)a*scy);\n"
    "    for (int c = 0; c < pixelsize; c++) {\n"
    "        float sum = 0.0f, wsum = 0.0f;\n"
    "        int y0 = (int)ceilf(sy-supy), y1 = (int)floorf(sy+supy);\n"
    "        int x0 = (int)ceilf(sx-supx), x1 = (int)floorf(sx+supx);\n"
    "        for (int j = y0; j <= y1; j++) {\n"
    "            float dy = (sy-j)/fmaxf(scy, 1.0f);\n"
    "            float wy = gpu_sinc(dy)*gpu_sinc(dy/a);\n"
    "            int cj = gpu_clamp(j, 0, in_h-1);\n"
    "            for (int i = x0; i <= x1; i++) {\n"
    "                float dx = (sx-i)/fmaxf(scx, 1.0f);\n"
    "                float wx = gpu_sinc(dx)*gpu_sinc(dx/a);\n"
    "                int ci = gpu_clamp(i, 0, in_w-1);\n"
    "                float w = wx*wy;\n"
    "                sum += input[cj*in_ls + ci*pixelsize + c] * w;\n"
    "                wsum += w;\n"
    "            }\n"
    "        }\n"
    "        if (wsum > 0) sum /= wsum;\n"
    "        output[oy*out_ls + ox*pixelsize + c] = (uchar)gpu_clamp((int)(sum+0.5f), "
    "0, 255);\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void resample_nearest(\n"
    "    const uchar *input, uchar *output,\n"
    "    int in_w, int in_h, int out_w, int out_h, int pixelsize,\n"
    "    float box_x0, float box_y0, float box_x1, float box_y1)\n"
    "{\n"
    "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int oy = gid / out_w, ox = gid % out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    int in_ls = in_w*pixelsize, out_ls = out_w*pixelsize;\n"
    "    int sx = gpu_clamp((int)(box_x0 + ox*(box_x1-box_x0)/out_w), 0, in_w-1);\n"
    "    int sy = gpu_clamp((int)(box_y0 + oy*(box_y1-box_y0)/out_h), 0, in_h-1);\n"
    "    for (int c = 0; c < pixelsize; c++)\n"
    "        output[oy*out_ls + ox*pixelsize + c] = input[sy*in_ls + sx*pixelsize + "
    "c];\n"
    "}\n"
    "\n"
    "/* ============ Color Conversion ============ */\n"
    "\n"
    "extern \"C\" __global__ void convert_rgb_to_l(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return;\n"
    "    int si = g*4;\n"
    "    out[g] = (uchar)gpu_clamp((int)(in[si]*0.299f + in[si+1]*0.587f + "
    "in[si+2]*0.114f + 0.5f), 0, 255);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_l_to_rgb(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return;\n"
    "    uchar v = in[g]; int o = g*4;\n"
    "    out[o]=v; out[o+1]=v; out[o+2]=v; out[o+3]=255;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_rgb_to_rgba(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return; int i=g*4;\n"
    "    out[i]=in[i]; out[i+1]=in[i+1]; out[i+2]=in[i+2]; out[i+3]=255;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_rgba_to_rgb(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return; int i=g*4;\n"
    "    out[i]=in[i]; out[i+1]=in[i+1]; out[i+2]=in[i+2]; out[i+3]=0;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_rgba_to_l(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return; int si=g*4;\n"
    "    out[g] = (uchar)gpu_clamp((int)(in[si]*0.299f + in[si+1]*0.587f + "
    "in[si+2]*0.114f + 0.5f), 0, 255);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_l_to_rgba(\n"
    "    const uchar *in, uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= w*h) return;\n"
    "    uchar v=in[g]; int o=g*4; out[o]=v; out[o+1]=v; out[o+2]=v; out[o+3]=255;\n"
    "}\n"
    "\n"
    "/* ============ Blend / Composite ============ */\n"
    "\n"
    "extern \"C\" __global__ void blend(\n"
    "    const uchar *im1, const uchar *im2, uchar *out,\n"
    "    int total, float alpha) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    float v = (float)im1[i]*(1.0f-alpha) + (float)im2[i]*alpha;\n"
    "    out[i] = (uchar)gpu_clamp((int)(v+0.5f), 0, 255);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void alpha_composite(\n"
    "    const uchar *im1, const uchar *im2, uchar *out, int npix) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    int i = g*4;\n"
    "    float sR=im2[i],sG=im2[i+1],sB=im2[i+2],sA=im2[i+3];\n"
    "    float dR=im1[i],dG=im1[i+1],dB=im1[i+2],dA=im1[i+3];\n"
    "    float sa=sA/255.f, da=dA/255.f;\n"
    "    float oa = sa + da*(1.f-sa);\n"
    "    if (oa < 1e-6f) { out[i]=out[i+1]=out[i+2]=out[i+3]=0; return; }\n"
    "    out[i]   = (uchar)gpu_clamp((int)((sR*sa + dR*da*(1.f-sa))/oa+.5f), 0, 255);\n"
    "    out[i+1] = (uchar)gpu_clamp((int)((sG*sa + dG*da*(1.f-sa))/oa+.5f), 0, 255);\n"
    "    out[i+2] = (uchar)gpu_clamp((int)((sB*sa + dB*da*(1.f-sa))/oa+.5f), 0, 255);\n"
    "    out[i+3] = (uchar)gpu_clamp((int)(oa*255.f+.5f), 0, 255);\n"
    "}\n"
    "\n"
    "/* ============ Channel Ops ============ */\n"
    "\n"
    "extern \"C\" __global__ void chop_operation(\n"
    "    const uchar *im1, const uchar *im2, uchar *out,\n"
    "    int total, int op, float scale, int off) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    int a = im1[i], b = im2[i], r;\n"
    "    switch(op) {\n"
    "        case 0: r=(int)((float)(a+b)/scale+off); break;\n"
    "        case 1: r=(int)((float)(a-b)/scale+off); break;\n"
    "        case 2: r=MULDIV255(a,b); break;\n"
    "        case 3: r=255-MULDIV255(255-a,255-b); break;\n"
    "        case 4: r=(a<128)?MULDIV255(2*a,b):255-MULDIV255(2*(255-a),255-b); "
    "break;\n"
    "        case 5: r=abs(a-b); break;\n"
    "        case 6: r=max(a,b); break;\n"
    "        case 7: r=min(a,b); break;\n"
    "        case 8: r=(a+b)&0xFF; break;\n"
    "        case 9: r=(a-b)&0xFF; break;\n"
    "        case 10: r=MULDIV255(a,a+MULDIV255(2*b,255-a)); break;\n"
    "        case 11: r=(b<128)?MULDIV255(2*b,a):255-MULDIV255(2*(255-b),255-a); "
    "break;\n"
    "        case 12: r=a&b; break;\n"
    "        case 13: r=a|b; break;\n"
    "        case 14: r=a^b; break;\n"
    "        case 15: r=255-a; break;\n"
    "        default: r=a;\n"
    "    }\n"
    "    out[i] = (uchar)gpu_clamp(r, 0, 255);\n"
    "}\n"
    "\n"
    "/* ============ Geometry ============ */\n"
    "\n"
    "extern \"C\" __global__ void transpose_op(\n"
    "    const uchar *in, uchar *out,\n"
    "    int in_w, int in_h, int out_w, int out_h, int ps, int op) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int oy = g/out_w, ox = g%out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    int sx, sy;\n"
    "    switch(op) {\n"
    "        case 0: sx=in_w-1-ox; sy=oy; break;\n"
    "        case 1: sx=ox; sy=in_h-1-oy; break;\n"
    "        case 2: sx=oy; sy=in_h-1-ox; break;\n"
    "        case 3: sx=in_w-1-ox; sy=in_h-1-oy; break;\n"
    "        case 4: sx=in_w-1-oy; sy=ox; break;\n"
    "        case 5: sx=oy; sy=ox; break;\n"
    "        case 6: sx=in_w-1-oy; sy=in_h-1-ox; break;\n"
    "        default: sx=ox; sy=oy;\n"
    "    }\n"
    "    int si = sy*in_w*ps + sx*ps;\n"
    "    int di = oy*out_w*ps + ox*ps;\n"
    "    for (int c=0; c<ps; c++) out[di+c] = in[si+c];\n"
    "}\n"
    "\n"
    "/* ============ Point / LUT ============ */\n"
    "\n"
    "extern \"C\" __global__ void point_lut(\n"
    "    const uchar *in, uchar *out, const uchar *lut,\n"
    "    int npix, int bands) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    if (bands == 1) { out[g] = lut[in[g]]; return; }\n"
    "    int base = g*4;\n"
    "    for (int b=0; b<bands && b<4; b++)\n"
    "        out[base+b] = lut[b*256 + in[base+b]];\n"
    "    for (int b=bands; b<4; b++)\n"
    "        out[base+b] = in[base+b];\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void point_transform(\n"
    "    const uchar *in, uchar *out, int total, float scale, float offset) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    out[i] = (uchar)gpu_clamp((int)((float)in[i]*scale + offset + 0.5f), 0, "
    "255);\n"
    "}\n"
    "\n"
    "/* ============ Band Ops ============ */\n"
    "\n"
    "extern \"C\" __global__ void getband(\n"
    "    const uchar *in, uchar *out, int npix, int ps, int band) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    out[g] = in[g*ps + band];\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void putband(\n"
    "    uchar *im, const uchar *band_data, int npix, int ps, int band) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    im[g*ps + band] = band_data[g];\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void fillband(\n"
    "    uchar *im, int npix, int ps, int band, int color) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    im[g*ps + band] = (uchar)color;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void fill_color(\n"
    "    uchar *im, int npix, int ps, int c0, int c1, int c2, int c3) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    int b = g*ps;\n"
    "    if (ps>=1) im[b]=(uchar)c0;\n"
    "    if (ps>=2) im[b+1]=(uchar)c1;\n"
    "    if (ps>=3) im[b+2]=(uchar)c2;\n"
    "    if (ps>=4) im[b+3]=(uchar)c3;\n"
    "}\n"
    "\n"
    "/* ============ Unsharp Mask ============ */\n"
    "\n"
    "extern \"C\" __global__ void unsharp_mask(\n"
    "    const uchar *orig, const uchar *blurred, uchar *out,\n"
    "    int total, int ps, int percent, int threshold) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    int diff = (int)orig[i] - (int)blurred[i];\n"
    "    if (abs(diff) >= threshold) {\n"
    "        int val = (int)orig[i] + diff * percent / 100;\n"
    "        out[i] = (uchar)gpu_clamp(val, 0, 255);\n"
    "    } else {\n"
    "        out[i] = orig[i];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Histogram (shared-memory reduction) ============ */\n"
    "\n"
    "extern \"C\" __global__ void histogram(\n"
    "    const uchar *in, int *hist, int npix, int ps, int bands) {\n"
    "    /* Local histogram per block, then atomic merge to global */\n"
    "    __shared__ int local_hist[1024]; /* 4 bands * 256 */\n"
    "    int tid = threadIdx.x;\n"
    "    /* Clear local histogram */\n"
    "    for (int i = tid; i < bands * 256; i += blockDim.x)\n"
    "        local_hist[i] = 0;\n"
    "    __syncthreads();\n"
    "    /* Accumulate into local histogram */\n"
    "    int g = blockIdx.x * blockDim.x + tid;\n"
    "    if (g < npix) {\n"
    "        if (ps == 1) {\n"
    "            atomicAdd(&local_hist[in[g]], 1);\n"
    "        } else {\n"
    "            int base = g * ps;\n"
    "            for (int b = 0; b < bands; b++)\n"
    "                atomicAdd(&local_hist[b * 256 + in[base + b]], 1);\n"
    "        }\n"
    "    }\n"
    "    __syncthreads();\n"
    "    /* Merge local histogram to global */\n"
    "    for (int i = tid; i < bands * 256; i += blockDim.x)\n"
    "        if (local_hist[i] > 0)\n"
    "            atomicAdd(&hist[i], local_hist[i]);\n"
    "}\n"
    "\n"
    "/* ============ Affine Transform ============ */\n"
    "\n"
    "extern \"C\" __global__ void affine_transform(\n"
    "    const uchar *in, uchar *out,\n"
    "    int in_w, int in_h, int out_w, int out_h, int ps,\n"
    "    float a0, float a1, float a2, float a3, float a4, float a5,\n"
    "    int filt, int fill) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int oy = g/out_w, ox = g%out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    float sx = a0*ox + a1*oy + a2;\n"
    "    float sy = a3*ox + a4*oy + a5;\n"
    "    int in_ls = in_w*ps, out_ls = out_w*ps;\n"
    "    int isx = (int)sx, isy = (int)sy;\n"
    "    if (isx >= 0 && isx < in_w && isy >= 0 && isy < in_h) {\n"
    "        for (int c=0; c<ps; c++)\n"
    "            out[oy*out_ls + ox*ps + c] = in[isy*in_ls + isx*ps + c];\n"
    "    } else if (fill) {\n"
    "        for (int c=0; c<ps; c++)\n"
    "            out[oy*out_ls + ox*ps + c] = 0;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Perspective Transform ============ */\n"
    "\n"
    "extern \"C\" __global__ void perspective_transform(\n"
    "    const uchar *in, uchar *out,\n"
    "    int in_w, int in_h, int out_w, int out_h, int ps,\n"
    "    float a0, float a1, float a2, float a3, float a4, float a5,\n"
    "    float a6, float a7, int filt, int fill) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int oy = g/out_w, ox = g%out_w;\n"
    "    if (ox >= out_w || oy >= out_h) return;\n"
    "    float denom = a6*ox + a7*oy + 1.0f;\n"
    "    if (fabsf(denom) < 1e-10f) denom = 1e-10f;\n"
    "    float sx = (a0*ox + a1*oy + a2) / denom;\n"
    "    float sy = (a3*ox + a4*oy + a5) / denom;\n"
    "    int in_ls = in_w*ps, out_ls = out_w*ps;\n"
    "    int isx = (int)sx, isy = (int)sy;\n"
    "    if (isx >= 0 && isx < in_w && isy >= 0 && isy < in_h) {\n"
    "        for (int c=0; c<ps; c++)\n"
    "            out[oy*out_ls + ox*ps + c] = in[isy*in_ls + isx*ps + c];\n"
    "    } else if (fill) {\n"
    "        for (int c=0; c<ps; c++)\n"
    "            out[oy*out_ls + ox*ps + c] = 0;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ CMYK conversions ============ */\n"
    "\n"
    "extern \"C\" __global__ void convert_rgb_to_cmyk(\n"
    "    const uchar *in, uchar *out, int npix) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= npix) return;\n"
    "    int base3 = i*4; int base4 = i*4;\n"
    "    float r = in[base3]/255.0f, g = in[base3+1]/255.0f, b = in[base3+2]/255.0f;\n"
    "    float k = 1.0f - fmaxf(fmaxf(r,g),b);\n"
    "    float inv = (k < 1.0f) ? 1.0f/(1.0f-k) : 0.0f;\n"
    "    out[base4]   = (uchar)gpu_clamp((int)((1.0f-r-k)*inv*255+0.5f),0,255);\n"
    "    out[base4+1] = (uchar)gpu_clamp((int)((1.0f-g-k)*inv*255+0.5f),0,255);\n"
    "    out[base4+2] = (uchar)gpu_clamp((int)((1.0f-b-k)*inv*255+0.5f),0,255);\n"
    "    out[base4+3] = (uchar)gpu_clamp((int)(k*255+0.5f),0,255);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_cmyk_to_rgb(\n"
    "    const uchar *in, uchar *out, int npix) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= npix) return;\n"
    "    int base = i*4;\n"
    "    float c2 = in[base]/255.0f, m = in[base+1]/255.0f;\n"
    "    float y = in[base+2]/255.0f, k = in[base+3]/255.0f;\n"
    "    out[base]   = (uchar)gpu_clamp((int)((1-c2)*(1-k)*255+0.5f),0,255);\n"
    "    out[base+1] = (uchar)gpu_clamp((int)((1-m)*(1-k)*255+0.5f),0,255);\n"
    "    out[base+2] = (uchar)gpu_clamp((int)((1-y)*(1-k)*255+0.5f),0,255);\n"
    "    out[base+3] = 255;\n"
    "}\n"
    "\n"
    "/* ============ Merge bands ============ */\n"
    "\n"
    "extern \"C\" __global__ void merge_bands(\n"
    "    const uchar *b0, const uchar *b1, const uchar *b2, const uchar *b3,\n"
    "    uchar *out, int npix, int nbands) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= npix) return;\n"
    "    int ps = (nbands <= 2) ? nbands : 4;\n"
    "    int base = i * ps;\n"
    "    out[base] = b0[i];\n"
    "    if (nbands >= 2) out[base+1] = b1[i];\n"
    "    if (nbands >= 3) out[base+2] = b2[i];\n"
    "    if (nbands >= 4) out[base+3] = b3[i];\n"
    "}\n"
    "\n"
    "/* ============ Paste with mask ============ */\n"
    "\n"
    "extern \"C\" __global__ void paste_with_mask(\n"
    "    uchar *dest, const uchar *src, const uchar *mask,\n"
    "    int dw, int dh, int sw, int sh, int ps, int dx, int dy) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int sy2 = g / sw, sx2 = g % sw;\n"
    "    if (sy2 >= sh) return;\n"
    "    int ox = sx2 + dx, oy = sy2 + dy;\n"
    "    if (ox < 0 || ox >= dw || oy < 0 || oy >= dh) return;\n"
    "    int alpha = mask[g];\n"
    "    int dst_off = oy * dw * ps + ox * ps;\n"
    "    int src_off = g * ps;\n"
    "    if (alpha == 255) {\n"
    "        for (int c=0; c<ps; c++) dest[dst_off+c] = src[src_off+c];\n"
    "    } else if (alpha > 0) {\n"
    "        for (int c=0; c<ps; c++) {\n"
    "            int d = dest[dst_off+c], s = src[src_off+c];\n"
    "            dest[dst_off+c] = (uchar)((s*alpha + d*(255-alpha) + 127)/255);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Color matrix ============ */\n"
    "\n"
    "extern \"C\" __global__ void color_matrix(\n"
    "    const uchar *in, uchar *out, int npix, int ps,\n"
    "    float m00, float m01, float m02, float m03,\n"
    "    float m10, float m11, float m12, float m13,\n"
    "    float m20, float m21, float m22, float m23,\n"
    "    float m30, float m31, float m32, float m33) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    int base = g*ps;\n"
    "    float r = (float)in[base];\n"
    "    float gc = (ps>=2) ? (float)in[base+1] : 0.0f;\n"
    "    float b = (ps>=3) ? (float)in[base+2] : 0.0f;\n"
    "    float a = (ps>=4) ? (float)in[base+3] : 255.0f;\n"
    "    float nr = m00*r + m01*gc + m02*b + m03*a;\n"
    "    float ng = m10*r + m11*gc + m12*b + m13*a;\n"
    "    float nb = m20*r + m21*gc + m22*b + m23*a;\n"
    "    float na = m30*r + m31*gc + m32*b + m33*a;\n"
    "    out[base] = (uchar)gpu_clamp((int)(nr+0.5f),0,255);\n"
    "    if (ps>=2) out[base+1] = (uchar)gpu_clamp((int)(ng+0.5f),0,255);\n"
    "    if (ps>=3) out[base+2] = (uchar)gpu_clamp((int)(nb+0.5f),0,255);\n"
    "    if (ps>=4) out[base+3] = (uchar)gpu_clamp((int)(na+0.5f),0,255);\n"
    "}\n"
    "\n"
    "/* ============ Crop ============ */\n"
    "\n"
    "extern \"C\" __global__ void crop_region(\n"
    "    const uchar *in, uchar *out,\n"
    "    int in_w, int out_w, int out_h, int ps, int x0, int y0) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int dy = g/out_w, dx = g%out_w;\n"
    "    if (dy >= out_h) return;\n"
    "    int src_off = (y0+dy)*in_w*ps + (x0+dx)*ps;\n"
    "    int dst_off = dy*out_w*ps + dx*ps;\n"
    "    for (int c=0; c<ps; c++) out[dst_off+c] = in[src_off+c];\n"
    "}\n"
    "\n"
    "/* ============ Expand ============ */\n"
    "\n"
    "extern \"C\" __global__ void expand_image(\n"
    "    const uchar *in, uchar *out,\n"
    "    int in_w, int in_h, int out_w, int out_h,\n"
    "    int ps, int pad_x, int pad_y,\n"
    "    int fill0, int fill1, int fill2, int fill3) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int dy = g/out_w, dx = g%out_w;\n"
    "    if (dy >= out_h) return;\n"
    "    int dst_off = dy*out_w*ps + dx*ps;\n"
    "    int sx = dx-pad_x, sy = dy-pad_y;\n"
    "    if (sx>=0 && sx<in_w && sy>=0 && sy<in_h) {\n"
    "        int src_off = sy*in_w*ps + sx*ps;\n"
    "        for (int c=0; c<ps; c++) out[dst_off+c] = in[src_off+c];\n"
    "    } else {\n"
    "        if (ps>=1) out[dst_off] = (uchar)fill0;\n"
    "        if (ps>=2) out[dst_off+1] = (uchar)fill1;\n"
    "        if (ps>=3) out[dst_off+2] = (uchar)fill2;\n"
    "        if (ps>=4) out[dst_off+3] = (uchar)fill3;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Offset ============ */\n"
    "\n"
    "extern \"C\" __global__ void offset_image(\n"
    "    const uchar *in, uchar *out,\n"
    "    int w, int h, int ps, int xoff, int yoff) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int y = g/w, x = g%w;\n"
    "    if (y >= h) return;\n"
    "    int sx = ((x-xoff)%w+w)%w;\n"
    "    int sy = ((y-yoff)%h+h)%h;\n"
    "    int dst_off = y*w*ps + x*ps;\n"
    "    int src_off = sy*w*ps + sx*ps;\n"
    "    for (int c=0; c<ps; c++) out[dst_off+c] = in[src_off+c];\n"
    "}\n"
    "\n"
    "/* ============ Negative ============ */\n"
    "\n"
    "extern \"C\" __global__ void negative(\n"
    "    const uchar *in, uchar *out, int total) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    out[i] = 255 - in[i];\n"
    "}\n"
    "\n"
    "/* ============ Posterize / Solarize ============ */\n"
    "\n"
    "extern \"C\" __global__ void posterize(\n"
    "    const uchar *in, uchar *out, int total, int bits) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    uchar mask = (uchar)(0xFF << (8-bits));\n"
    "    out[i] = in[i] & mask;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void solarize(\n"
    "    const uchar *in, uchar *out, int total, int thresh) {\n"
    "    int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (i >= total) return;\n"
    "    uchar v = in[i];\n"
    "    out[i] = (v >= thresh) ? (255-v) : v;\n"
    "}\n"
    "\n"
    "/* ============ Equalize ============ */\n"
    "\n"
    "extern \"C\" __global__ void equalize(\n"
    "    const uchar *in, uchar *out, const uchar *lut,\n"
    "    int npix, int ps, int bands) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (g >= npix) return;\n"
    "    int base = g*ps;\n"
    "    for (int b2=0; b2<bands; b2++)\n"
    "        out[base+b2] = lut[b2*256 + in[base+b2]];\n"
    "    if (ps > bands)\n"
    "        out[base+ps-1] = in[base+ps-1];\n"
    "}\n"
    "\n"
    "/* ============ Gradient ============ */\n"
    "\n"
    "extern \"C\" __global__ void linear_gradient(\n"
    "    uchar *out, int w, int h, int dir) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int y = g/w, x = g%w;\n"
    "    if (y >= h) return;\n"
    "    out[g] = (dir==0)\n"
    "        ? (uchar)(y*255/(h>1?h-1:1))\n"
    "        : (uchar)(x*255/(w>1?w-1:1));\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void radial_gradient(\n"
    "    uchar *out, int w, int h) {\n"
    "    int g = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    int y = g/w, x = g%w;\n"
    "    if (y >= h) return;\n"
    "    float cx = (float)w*0.5f, cy = (float)h*0.5f;\n"
    "    float fdx = (float)x-cx, fdy = (float)y-cy;\n"
    "    float maxr = sqrtf(cx*cx+cy*cy);\n"
    "    float r = sqrtf(fdx*fdx+fdy*fdy);\n"
    "    out[g] = (uchar)gpu_clamp((int)(r/maxr*255.0f+0.5f),0,255);\n"
    "}\n"
    "\n"
    "/* ============ Getbbox (parallel reduction via atomicMin/Max) ============ */\n"
    "\n"
    "extern \"C\" __global__ void getbbox(\n"
    "    const uchar *input, int *result,\n"
    "    int width, int height, int pixelsize, int check_channel) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= width*height) return;\n"
    "    int y = gid/width, x = gid%width;\n"
    "    int base = gid*pixelsize;\n"
    "    int nonzero = 0;\n"
    "    if (check_channel >= 0) { nonzero = (input[base+check_channel] != 0); }\n"
    "    else { for(int c=0;c<pixelsize;c++) if(input[base+c]!=0){nonzero=1;break;} }\n"
    "    if (nonzero) {\n"
    "        atomicMin(&result[0], x);\n"
    "        atomicMin(&result[1], y);\n"
    "        atomicMax(&result[2], x);\n"
    "        atomicMax(&result[3], y);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Getextrema (parallel min/max per band) ============ */\n"
    "\n"
    "extern \"C\" __global__ void getextrema(\n"
    "    const uchar *input, int *result,\n"
    "    int num_pixels, int pixelsize, int bands) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= num_pixels) return;\n"
    "    int base = gid*pixelsize;\n"
    "    for (int b=0;b<bands;b++) {\n"
    "        int v = (int)input[base+b];\n"
    "        atomicMin(&result[b*2], v);\n"
    "        atomicMax(&result[b*2+1], v);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Effect spread (hash-based PRNG) ============ */\n"
    "\n"
    "extern \"C\" __global__ void effect_spread(\n"
    "    const uchar *input, uchar *output,\n"
    "    int width, int height, int pixelsize, int distance, unsigned int seed) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= width*height) return;\n"
    "    int y = gid/width, x = gid%width;\n"
    "    unsigned int h = (unsigned int)gid ^ seed;\n"
    "    h = (h + 0x7ed55d16u) + (h << 12);\n"
    "    h = (h ^ 0xc761c23cu) ^ (h >> 19);\n"
    "    h = (h + 0x165667b1u) + (h << 5);\n"
    "    h = (h + 0xd3a2646cu) ^ (h << 9);\n"
    "    h = (h + 0xfd7046c5u) + (h << 3);\n"
    "    h = (h ^ 0xb55a4f09u) ^ (h >> 16);\n"
    "    int diam = 2*distance+1;\n"
    "    int dx = (int)(h % (unsigned int)diam) - distance;\n"
    "    h = h * 2654435761u + 0x12345u;\n"
    "    int dy = (int)(h % (unsigned int)diam) - distance;\n"
    "    int sx = gpu_clamp(x+dx, 0, width-1);\n"
    "    int sy = gpu_clamp(y+dy, 0, height-1);\n"
    "    int src = (sy*width+sx)*pixelsize;\n"
    "    int dst = gid*pixelsize;\n"
    "    for(int c=0;c<pixelsize;c++) output[dst+c] = input[src+c];\n"
    "}\n"
    "\n"
    "/* ============ YCbCr conversion ============ */\n"
    "\n"
    "extern \"C\" __global__ void convert_ycbcr_to_rgb(\n"
    "    const uchar *input, uchar *output,\n"
    "    int num_pixels, int in_ps, int out_ps) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= num_pixels) return;\n"
    "    int ib = gid*in_ps, ob = gid*out_ps;\n"
    "    float yv = (float)input[ib];\n"
    "    float cb = (float)input[ib+1]-128.0f;\n"
    "    float cr = (float)input[ib+2]-128.0f;\n"
    "    output[ob]   = (uchar)gpu_clamp((int)(yv+1.402f*cr+0.5f),0,255);\n"
    "    output[ob+1] = "
    "(uchar)gpu_clamp((int)(yv-0.344136f*cb-0.714136f*cr+0.5f),0,255);\n"
    "    output[ob+2] = (uchar)gpu_clamp((int)(yv+1.772f*cb+0.5f),0,255);\n"
    "    if(out_ps>3) output[ob+3]=255;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void convert_rgb_to_ycbcr(\n"
    "    const uchar *input, uchar *output,\n"
    "    int num_pixels, int in_ps, int out_ps) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= num_pixels) return;\n"
    "    int ib = gid*in_ps, ob = gid*out_ps;\n"
    "    float r=(float)input[ib], g=(float)input[ib+1], b=(float)input[ib+2];\n"
    "    output[ob]   = "
    "(uchar)gpu_clamp((int)(0.299f*r+0.587f*g+0.114f*b+0.5f),0,255);\n"
    "    output[ob+1] = "
    "(uchar)gpu_clamp((int)(-0.168736f*r-0.331264f*g+0.5f*b+128.5f),0,255);\n"
    "    output[ob+2] = "
    "(uchar)gpu_clamp((int)(0.5f*r-0.418688f*g-0.081312f*b+128.5f),0,255);\n"
    "    if(out_ps>3) output[ob+3]=255;\n"
    "}\n"
    "\n"
    "/* ============ Reduce (integer downscale) ============ */\n"
    "\n"
    "extern \"C\" __global__ void reduce_image(\n"
    "    const uchar *input, uchar *output,\n"
    "    int in_w, int in_h, int out_w, int out_h,\n"
    "    int factor_x, int factor_y, int pixelsize, int bands) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= out_w*out_h) return;\n"
    "    int oy = gid/out_w, ox = gid%out_w;\n"
    "    int in_ls = in_w*pixelsize, out_ls = out_w*pixelsize;\n"
    "    for (int b2=0;b2<bands;b2++) {\n"
    "        int acc=0, count=0;\n"
    "        for(int dy=0;dy<factor_y;dy++) {\n"
    "            int iy=oy*factor_y+dy; if(iy>=in_h) break;\n"
    "            for(int dx=0;dx<factor_x;dx++) {\n"
    "                int ix=ox*factor_x+dx; if(ix>=in_w) break;\n"
    "                acc+=(int)input[iy*in_ls+ix*pixelsize+b2]; count++;\n"
    "            }\n"
    "        }\n"
    "        output[oy*out_ls+ox*pixelsize+b2] = (uchar)((acc+count/2)/count);\n"
    "    }\n"
    "    if(pixelsize>bands) {\n"
    "        int iy=oy*factor_y, ix=ox*factor_x;\n"
    "        if(iy<in_h&&ix<in_w)\n"
    "            output[oy*out_ls+ox*pixelsize+pixelsize-1] =\n"
    "                input[iy*in_ls+ix*pixelsize+pixelsize-1];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============ Rank filter ============ */\n"
    "\n"
    "extern \"C\" __global__ void rank_filter(\n"
    "    const uchar *input, uchar *output,\n"
    "    int width, int height, int pixelsize, int bands,\n"
    "    int ksize, int rank) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= width*height) return;\n"
    "    int y = gid/width, x = gid%width;\n"
    "    int half = ksize/2, ls = width*pixelsize;\n"
    "    for (int b2=0;b2<bands;b2++) {\n"
    "        uchar vals[49];\n"
    "        int count=0;\n"
    "        for(int dy=-half;dy<=half;dy++) {\n"
    "            int iy=gpu_clamp(y+dy,0,height-1);\n"
    "            for(int dx=-half;dx<=half;dx++) {\n"
    "                int ix=gpu_clamp(x+dx,0,width-1);\n"
    "                vals[count++]=input[iy*ls+ix*pixelsize+b2];\n"
    "            }\n"
    "        }\n"
    "        for(int i=0;i<=rank&&i<count;i++) {\n"
    "            int mi=i;\n"
    "            for(int j=i+1;j<count;j++) if(vals[j]<vals[mi]) mi=j;\n"
    "            uchar tmp=vals[i]; vals[i]=vals[mi]; vals[mi]=tmp;\n"
    "        }\n"
    "        output[y*ls+x*pixelsize+b2] = vals[rank];\n"
    "    }\n"
    "    if(pixelsize>bands)\n"
    "        output[y*ls+x*pixelsize+pixelsize-1] = "
    "input[y*ls+x*pixelsize+pixelsize-1];\n"
    "}\n"
    "\n"
    "/* ============ Mode filter ============ */\n"
    "\n"
    "extern \"C\" __global__ void mode_filter(\n"
    "    const uchar *input, uchar *output,\n"
    "    int width, int height, int ksize) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= width*height) return;\n"
    "    int y = gid/width, x = gid%width;\n"
    "    int half = ksize/2;\n"
    "    uchar hist[256];\n"
    "    for(int i=0;i<256;i++) hist[i]=0;\n"
    "    for(int dy=-half;dy<=half;dy++) {\n"
    "        int iy=gpu_clamp(y+dy,0,height-1);\n"
    "        for(int dx=-half;dx<=half;dx++) {\n"
    "            int ix=gpu_clamp(x+dx,0,width-1);\n"
    "            hist[input[iy*width+ix]]++;\n"
    "        }\n"
    "    }\n"
    "    uchar best_val=input[y*width+x], best_count=1;\n"
    "    for(int i=0;i<256;i++) "
    "if(hist[i]>best_count){best_count=hist[i];best_val=(uchar)i;}\n"
    "    output[y*width+x] = best_val;\n"
    "}\n"
    "\n"
    "/* ============ Color LUT 3D ============ */\n"
    "\n"
    "extern \"C\" __global__ void color_lut_3d(\n"
    "    const uchar *input, uchar *output,\n"
    "    const short *table, int npix, int pixelsize,\n"
    "    int table_channels, int size1D, int size2D, int size3D) {\n"
    "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
    "    if (gid >= npix) return;\n"
    "    int base = gid*pixelsize;\n"
    "    float r = input[base] / 255.0f * (size1D-1);\n"
    "    float g = input[base+1] / 255.0f * (size2D-1);\n"
    "    float b = input[base+2] / 255.0f * (size3D-1);\n"
    "    int r0=(int)r, g0=(int)g, b0=(int)b;\n"
    "    int r1=min(r0+1,size1D-1), g1=min(g0+1,size2D-1), b1=min(b0+1,size3D-1);\n"
    "    float fr=r-r0, fg=g-g0, fb=b-b0;\n"
    "    for(int c=0;c<table_channels&&c<pixelsize;c++) {\n"
    "        float c000=table[((r0*size2D+g0)*size3D+b0)*table_channels+c];\n"
    "        float c001=table[((r0*size2D+g0)*size3D+b1)*table_channels+c];\n"
    "        float c010=table[((r0*size2D+g1)*size3D+b0)*table_channels+c];\n"
    "        float c011=table[((r0*size2D+g1)*size3D+b1)*table_channels+c];\n"
    "        float c100=table[((r1*size2D+g0)*size3D+b0)*table_channels+c];\n"
    "        float c101=table[((r1*size2D+g0)*size3D+b1)*table_channels+c];\n"
    "        float c110=table[((r1*size2D+g1)*size3D+b0)*table_channels+c];\n"
    "        float c111=table[((r1*size2D+g1)*size3D+b1)*table_channels+c];\n"
    "        float val=c000*(1-fr)*(1-fg)*(1-fb)+c001*(1-fr)*(1-fg)*fb\n"
    "            +c010*(1-fr)*fg*(1-fb)+c011*(1-fr)*fg*fb\n"
    "            +c100*fr*(1-fg)*(1-fb)+c101*fr*(1-fg)*fb\n"
    "            +c110*fr*fg*(1-fb)+c111*fr*fg*fb;\n"
    "        output[base+c]=(uchar)gpu_clamp((int)(val/255.0f+0.5f),0,255);\n"
    "    }\n"
    "    if(pixelsize==4&&table_channels<4) output[base+3]=input[base+3];\n"
    "}\n";

/* ================================================================== */
/* CUDA context state                                                   */
/* ================================================================== */

typedef struct {
    CUcontext cu_context;
    CUdevice cu_device;
    CUmodule cu_module;

    /* Kernel function handles */
    CUfunction f_box_blur_h;
    CUfunction f_box_blur_v;
    CUfunction f_gaussian_blur_h;
    CUfunction f_gaussian_blur_v;
    CUfunction f_convolve;
    CUfunction f_resample_nearest;
    CUfunction f_resample_bilinear;
    CUfunction f_resample_bicubic;
    CUfunction f_resample_lanczos;
    CUfunction f_convert_rgb_to_l;
    CUfunction f_convert_l_to_rgb;
    CUfunction f_convert_rgb_to_rgba;
    CUfunction f_convert_rgba_to_rgb;
    CUfunction f_convert_rgba_to_l;
    CUfunction f_convert_l_to_rgba;
    CUfunction f_blend;
    CUfunction f_alpha_composite;
    CUfunction f_chop_operation;
    CUfunction f_transpose_op;
    CUfunction f_point_lut;
    CUfunction f_point_transform;
    CUfunction f_getband;
    CUfunction f_putband;
    CUfunction f_fillband;
    CUfunction f_fill_color;
    CUfunction f_unsharp_mask;
    CUfunction f_histogram;
    CUfunction f_affine_transform;
    CUfunction f_perspective_transform;
    CUfunction f_convert_rgb_to_cmyk;
    CUfunction f_convert_cmyk_to_rgb;
    CUfunction f_merge_bands;
    CUfunction f_paste_with_mask;
    CUfunction f_color_matrix;
    CUfunction f_crop_region;
    CUfunction f_expand_image;
    CUfunction f_offset_image;
    CUfunction f_negative;
    CUfunction f_posterize;
    CUfunction f_solarize;
    CUfunction f_equalize;
    CUfunction f_linear_gradient;
    CUfunction f_radial_gradient;
    CUfunction f_getbbox;
    CUfunction f_getextrema;
    CUfunction f_effect_spread;
    CUfunction f_convert_ycbcr_to_rgb;
    CUfunction f_convert_rgb_to_ycbcr;
    CUfunction f_reduce;
    CUfunction f_rank_filter;
    CUfunction f_mode_filter;
    CUfunction f_color_lut_3d_gpu;
} CUDAContext;

/* ================================================================== */
/* Helpers                                                              */
/* ================================================================== */

#define CUDA_BLOCK_SIZE 256

static unsigned int
_cuda_grid(size_t total) {
    return (unsigned int)((total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
}

static CUfunction
_cuda_get_func(CUmodule mod, const char *name) {
    CUfunction f;
    CUresult err = cuModuleGetFunction(&f, mod, name);
    if (err != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "Pillow GPU CUDA: failed to get function '%s' (err=%d)\n",
            name,
            (int)err
        );
        return NULL;
    }
    return f;
}

/* ================================================================== */
/* Backend operation implementations                                    */
/* ================================================================== */

static void
_cuda_shutdown(GPUBackend self) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (ctx) {
        if (ctx->cu_module)
            cuModuleUnload(ctx->cu_module);
        if (ctx->cu_context)
            cuCtxDestroy(ctx->cu_context);
        free(ctx);
    }
    free(self);
}

static int
_cuda_buffer_alloc(GPUBackend self, GPUBuffer buf, size_t size) {
    (void)self;
    CUdeviceptr ptr;
    CUresult err = cuMemAlloc(&ptr, size);
    if (err != CUDA_SUCCESS)
        return GPU_ERROR_MEMORY;
    buf->handle.cu_ptr = (unsigned long long)ptr;
    buf->size = size;
    return GPU_OK;
}

static void
_cuda_buffer_free(GPUBackend self, GPUBuffer buf) {
    (void)self;
    if (buf->handle.cu_ptr) {
        cuMemFree((CUdeviceptr)buf->handle.cu_ptr);
        buf->handle.cu_ptr = 0;
        buf->size = 0;
    }
}

static int
_cuda_buffer_upload(GPUBackend self, GPUBuffer buf, const void *data, size_t size) {
    (void)self;
    CUresult err = cuMemcpyHtoD((CUdeviceptr)buf->handle.cu_ptr, data, size);
    return (err == CUDA_SUCCESS) ? GPU_OK : GPU_ERROR_TRANSFER;
}

static int
_cuda_buffer_download(GPUBackend self, GPUBuffer buf, void *data, size_t size) {
    (void)self;
    CUresult err = cuMemcpyDtoH(data, (CUdeviceptr)buf->handle.cu_ptr, size);
    return (err == CUDA_SUCCESS) ? GPU_OK : GPU_ERROR_TRANSFER;
}

static int
_cuda_buffer_copy(GPUBackend self, GPUBuffer dst, GPUBuffer src, size_t size) {
    (void)self;
    CUresult err = cuMemcpyDtoD(
        (CUdeviceptr)dst->handle.cu_ptr, (CUdeviceptr)src->handle.cu_ptr, size
    );
    return (err == CUDA_SUCCESS) ? GPU_OK : GPU_ERROR_TRANSFER;
}

/* Helper to launch a kernel with args */
static int
_cuda_launch(CUfunction f, unsigned int grid, void **args) {
    CUresult err =
        cuLaunchKernel(f, grid, 1, 1, CUDA_BLOCK_SIZE, 1, 1, 0, NULL, args, NULL);
    if (err != CUDA_SUCCESS)
        return GPU_ERROR_LAUNCH;
    err = cuCtxSynchronize();
    return (err == CUDA_SUCCESS) ? GPU_OK : GPU_ERROR_LAUNCH;
}

/* -------------------------------------------------------------------- */
/* Blur                                                                  */
/* -------------------------------------------------------------------- */

static int
_cuda_box_blur(
    GPUBackend self, ImagingGPU out, ImagingGPU in, float xradius, float yradius, int n
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    int rx = (int)(xradius + 0.5f);
    int ry = (int)(yradius + 0.5f);
    if (rx < 0)
        rx = 0;
    if (ry < 0)
        ry = 0;

    /* Temp buffer */
    GPUBufferInstance tmp = {0};
    int err = _cuda_buffer_alloc(self, &tmp, in->buffer.size);
    if (err != GPU_OK)
        return err;

    _cuda_buffer_copy(self, &out->buffer, &in->buffer, in->buffer.size);

    CUdeviceptr out_ptr = (CUdeviceptr)out->buffer.handle.cu_ptr;
    CUdeviceptr tmp_ptr = (CUdeviceptr)tmp.handle.cu_ptr;
    int w = in->xsize, h = in->ysize, ps = in->pixelsize;

    for (int pass = 0; pass < n; pass++) {
        /* H: out -> tmp (per-pixel) */
        size_t total = (size_t)w * h;
        void *args_h[] = {&out_ptr, &tmp_ptr, &w, &h, &ps, &rx};
        err = _cuda_launch(ctx->f_box_blur_h, _cuda_grid(total), args_h);
        if (err != GPU_OK) {
            _cuda_buffer_free(self, &tmp);
            return err;
        }

        /* V: tmp -> out (per-pixel) */
        void *args_v[] = {&tmp_ptr, &out_ptr, &w, &h, &ps, &ry};
        err = _cuda_launch(ctx->f_box_blur_v, _cuda_grid(total), args_v);
        if (err != GPU_OK) {
            _cuda_buffer_free(self, &tmp);
            return err;
        }
    }

    _cuda_buffer_free(self, &tmp);
    return GPU_OK;
}

static int
_cuda_gaussian_blur(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    float xradius,
    float yradius,
    int passes
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;

    int rx = (int)ceilf(xradius * 2.57f);
    int ry = (int)ceilf(yradius * 2.57f);
    if (rx < 1)
        rx = 1;
    if (ry < 1)
        ry = 1;
    int xdiam = 2 * rx + 1, ydiam = 2 * ry + 1;

    float *xw = (float *)malloc(xdiam * sizeof(float));
    float *yw = (float *)malloc(ydiam * sizeof(float));
    if (!xw || !yw) {
        free(xw);
        free(yw);
        return GPU_ERROR_MEMORY;
    }

    float xs = 0, ys = 0;
    for (int i = 0; i < xdiam; i++) {
        float d = (float)(i - rx);
        xw[i] = expf(-d * d / (2.0f * xradius * xradius));
        xs += xw[i];
    }
    for (int i = 0; i < ydiam; i++) {
        float d = (float)(i - ry);
        yw[i] = expf(-d * d / (2.0f * yradius * yradius));
        ys += yw[i];
    }
    for (int i = 0; i < xdiam; i++) xw[i] /= xs;
    for (int i = 0; i < ydiam; i++) yw[i] /= ys;

    /* Upload weights */
    CUdeviceptr xw_d, yw_d;
    cuMemAlloc(&xw_d, xdiam * sizeof(float));
    cuMemAlloc(&yw_d, ydiam * sizeof(float));
    cuMemcpyHtoD(xw_d, xw, xdiam * sizeof(float));
    cuMemcpyHtoD(yw_d, yw, ydiam * sizeof(float));
    free(xw);
    free(yw);

    GPUBufferInstance tmp = {0};
    int err = _cuda_buffer_alloc(self, &tmp, in->buffer.size);
    if (err != GPU_OK) {
        cuMemFree(xw_d);
        cuMemFree(yw_d);
        return err;
    }

    CUdeviceptr out_ptr = (CUdeviceptr)out->buffer.handle.cu_ptr;
    CUdeviceptr in_ptr = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr tmp_ptr = (CUdeviceptr)tmp.handle.cu_ptr;
    int w = in->xsize, h = in->ysize, ps = in->pixelsize;
    size_t total = (size_t)w * h;

    _cuda_buffer_copy(self, &out->buffer, &in->buffer, in->buffer.size);

    for (int pass = 0; pass < passes; pass++) {
        void *ah[] = {&out_ptr, &tmp_ptr, &xw_d, &w, &h, &ps, &rx};
        _cuda_launch(ctx->f_gaussian_blur_h, _cuda_grid(total), ah);

        void *av[] = {&tmp_ptr, &out_ptr, &yw_d, &w, &h, &ps, &ry};
        _cuda_launch(ctx->f_gaussian_blur_v, _cuda_grid(total), av);
    }

    cuMemFree(xw_d);
    cuMemFree(yw_d);
    _cuda_buffer_free(self, &tmp);
    return GPU_OK;
}

static int
_cuda_unsharp_mask(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    float radius,
    int percent,
    int threshold
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;

    ImagingGPU blurred = ImagingGPU_NewDirty(in->mode, in->xsize, in->ysize);
    if (!blurred)
        return GPU_ERROR_MEMORY;

    int err = _cuda_gaussian_blur(self, blurred, in, radius, radius, 3);
    if (err != GPU_OK) {
        ImagingGPU_Delete(blurred);
        return err;
    }

    CUdeviceptr orig = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr blur = (CUdeviceptr)blurred->buffer.handle.cu_ptr;
    CUdeviceptr outp = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int total = (int)in->buffer.size;
    int ps = in->pixelsize;

    void *args[] = {&orig, &blur, &outp, &total, &ps, &percent, &threshold};
    err = _cuda_launch(ctx->f_unsharp_mask, _cuda_grid(total), args);
    ImagingGPU_Delete(blurred);
    return err;
}

/* -------------------------------------------------------------------- */
/* Filter                                                                */
/* -------------------------------------------------------------------- */

static int
_cuda_filter(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    int ksize_x,
    int ksize_y,
    const float *kernel,
    float divisor,
    float offset
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    size_t ksz = ksize_x * ksize_y * sizeof(float);
    CUdeviceptr k_d;
    cuMemAlloc(&k_d, ksz);
    cuMemcpyHtoD(k_d, kernel, ksz);

    CUdeviceptr inp = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr outp = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int w = in->xsize, h = in->ysize, ps = in->pixelsize;
    size_t total = (size_t)w * h;

    void *args[] = {
        &inp, &outp, &k_d, &w, &h, &ps, &ksize_x, &ksize_y, &divisor, &offset
    };
    int err = _cuda_launch(ctx->f_convolve, _cuda_grid(total), args);
    cuMemFree(k_d);
    return err;
}

/* -------------------------------------------------------------------- */
/* Resample                                                              */
/* -------------------------------------------------------------------- */

static int
_cuda_resample(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    int xsize,
    int ysize,
    int filter,
    const float box[4]
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUfunction f;
    switch (filter) {
        case GPU_RESAMPLE_NEAREST:
            f = ctx->f_resample_nearest;
            break;
        case GPU_RESAMPLE_BILINEAR:
            f = ctx->f_resample_bilinear;
            break;
        case GPU_RESAMPLE_BICUBIC:
            f = ctx->f_resample_bicubic;
            break;
        case GPU_RESAMPLE_LANCZOS:
            f = ctx->f_resample_lanczos;
            break;
        default:
            f = ctx->f_resample_bilinear;
    }

    CUdeviceptr inp = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr outp = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int iw = in->xsize, ih = in->ysize, ps = in->pixelsize;
    float bx0 = box[0], by0 = box[1], bx1 = box[2], by1 = box[3];
    size_t total = (size_t)xsize * ysize;

    void *args[] = {&inp, &outp, &iw, &ih, &xsize, &ysize, &ps, &bx0, &by0, &bx1, &by1};
    return _cuda_launch(f, _cuda_grid(total), args);
}

/* -------------------------------------------------------------------- */
/* Convert                                                               */
/* -------------------------------------------------------------------- */

static int
_cuda_convert(GPUBackend self, ImagingGPU out, ImagingGPU in, ModeID to_mode) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUfunction f = NULL;

    if (in->mode == IMAGING_MODE_RGB && to_mode == IMAGING_MODE_L)
        f = ctx->f_convert_rgb_to_l;
    else if (in->mode == IMAGING_MODE_L && to_mode == IMAGING_MODE_RGB)
        f = ctx->f_convert_l_to_rgb;
    else if (in->mode == IMAGING_MODE_RGB && to_mode == IMAGING_MODE_RGBA)
        f = ctx->f_convert_rgb_to_rgba;
    else if (in->mode == IMAGING_MODE_RGBA && to_mode == IMAGING_MODE_RGB)
        f = ctx->f_convert_rgba_to_rgb;
    else if (in->mode == IMAGING_MODE_RGBA && to_mode == IMAGING_MODE_L)
        f = ctx->f_convert_rgba_to_l;
    else if (in->mode == IMAGING_MODE_L && to_mode == IMAGING_MODE_RGBA)
        f = ctx->f_convert_l_to_rgba;
    else if (
        (in->mode == IMAGING_MODE_YCbCr) &&
        (to_mode == IMAGING_MODE_RGB || to_mode == IMAGING_MODE_RGBA)
    ) {
        CUfunction kk = ctx->f_convert_ycbcr_to_rgb;
        if (!kk)
            return GPU_ERROR_UNSUPPORTED;
        CUdeviceptr inp2 = (CUdeviceptr)in->buffer.handle.cu_ptr;
        CUdeviceptr outp2 = (CUdeviceptr)out->buffer.handle.cu_ptr;
        int npix = in->xsize * in->ysize;
        int in_ps = in->pixelsize, out_ps = out->pixelsize;
        void *args2[] = {&inp2, &outp2, &npix, &in_ps, &out_ps};
        return _cuda_launch(kk, _cuda_grid(npix), args2);
    } else if (
        (in->mode == IMAGING_MODE_RGB || in->mode == IMAGING_MODE_RGBA) &&
        to_mode == IMAGING_MODE_YCbCr
    ) {
        CUfunction kk = ctx->f_convert_rgb_to_ycbcr;
        if (!kk)
            return GPU_ERROR_UNSUPPORTED;
        CUdeviceptr inp2 = (CUdeviceptr)in->buffer.handle.cu_ptr;
        CUdeviceptr outp2 = (CUdeviceptr)out->buffer.handle.cu_ptr;
        int npix = in->xsize * in->ysize;
        int in_ps = in->pixelsize, out_ps = out->pixelsize;
        void *args2[] = {&inp2, &outp2, &npix, &in_ps, &out_ps};
        return _cuda_launch(kk, _cuda_grid(npix), args2);
    } else
        return GPU_ERROR_UNSUPPORTED;

    CUdeviceptr inp = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr outp = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int w = in->xsize, h = in->ysize;
    size_t total = (size_t)w * h;

    void *args[] = {&inp, &outp, &w, &h};
    return _cuda_launch(f, _cuda_grid(total), args);
}

/* -------------------------------------------------------------------- */
/* Blend / composite / chops / geometry / point / bands                  */
/* -------------------------------------------------------------------- */

static int
_cuda_blend(
    GPUBackend self, ImagingGPU out, ImagingGPU im1, ImagingGPU im2, float alpha
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr p1 = (CUdeviceptr)im1->buffer.handle.cu_ptr;
    CUdeviceptr p2 = (CUdeviceptr)im2->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int total = (int)im1->buffer.size;
    void *args[] = {&p1, &p2, &po, &total, &alpha};
    return _cuda_launch(ctx->f_blend, _cuda_grid(total), args);
}

static int
_cuda_alpha_composite(GPUBackend self, ImagingGPU out, ImagingGPU im1, ImagingGPU im2) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr p1 = (CUdeviceptr)im1->buffer.handle.cu_ptr;
    CUdeviceptr p2 = (CUdeviceptr)im2->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int npix = im1->xsize * im1->ysize;
    void *args[] = {&p1, &p2, &po, &npix};
    return _cuda_launch(ctx->f_alpha_composite, _cuda_grid(npix), args);
}

static int
_cuda_chop(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU im1,
    ImagingGPU im2,
    int op,
    float scale,
    int offset
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr p1 = (CUdeviceptr)im1->buffer.handle.cu_ptr;
    CUdeviceptr p2 = (CUdeviceptr)im2->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int total = (int)im1->buffer.size;
    void *args[] = {&p1, &p2, &po, &total, &op, &scale, &offset};
    return _cuda_launch(ctx->f_chop_operation, _cuda_grid(total), args);
}

static int
_cuda_transpose(GPUBackend self, ImagingGPU out, ImagingGPU in, int op) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int iw = in->xsize, ih = in->ysize, ow = out->xsize, oh = out->ysize,
        ps = in->pixelsize;
    size_t total = (size_t)ow * oh;
    void *args[] = {&pi, &po, &iw, &ih, &ow, &oh, &ps, &op};
    return _cuda_launch(ctx->f_transpose_op, _cuda_grid(total), args);
}

static int
_cuda_point_lut(
    GPUBackend self, ImagingGPU out, ImagingGPU in, const UINT8 *lut, int bands
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr lut_d;
    cuMemAlloc(&lut_d, bands * 256);
    cuMemcpyHtoD(lut_d, lut, bands * 256);
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int npix = in->xsize * in->ysize;
    void *args[] = {&pi, &po, &lut_d, &npix, &bands};
    int err = _cuda_launch(ctx->f_point_lut, _cuda_grid(npix), args);
    cuMemFree(lut_d);
    return err;
}

static int
_cuda_point_transform(
    GPUBackend self, ImagingGPU out, ImagingGPU in, double scale, double offset
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int total = (int)in->buffer.size;
    float fs = (float)scale, fo = (float)offset;
    void *args[] = {&pi, &po, &total, &fs, &fo};
    return _cuda_launch(ctx->f_point_transform, _cuda_grid(total), args);
}

static int
_cuda_getband(GPUBackend self, ImagingGPU out, ImagingGPU in, int band) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int npix = in->xsize * in->ysize, ps = in->pixelsize;
    void *args[] = {&pi, &po, &npix, &ps, &band};
    return _cuda_launch(ctx->f_getband, _cuda_grid(npix), args);
}

static int
_cuda_putband(GPUBackend self, ImagingGPU im, ImagingGPU band_im, int band) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    CUdeviceptr pb = (CUdeviceptr)band_im->buffer.handle.cu_ptr;
    int npix = im->xsize * im->ysize, ps = im->pixelsize;
    void *args[] = {&pi, &pb, &npix, &ps, &band};
    return _cuda_launch(ctx->f_putband, _cuda_grid(npix), args);
}

static int
_cuda_fillband(GPUBackend self, ImagingGPU im, int band, int color) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    int npix = im->xsize * im->ysize, ps = im->pixelsize;
    void *args[] = {&pi, &npix, &ps, &band, &color};
    return _cuda_launch(ctx->f_fillband, _cuda_grid(npix), args);
}

static int
_cuda_fill(GPUBackend self, ImagingGPU im, const void *color) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    int npix = im->xsize * im->ysize, ps = im->pixelsize;
    const UINT8 *c = (const UINT8 *)color;
    int c0 = c[0], c1 = (ps >= 2) ? c[1] : 0, c2 = (ps >= 3) ? c[2] : 0,
        c3 = (ps >= 4) ? c[3] : 0;
    void *args[] = {&pi, &npix, &ps, &c0, &c1, &c2, &c3};
    return _cuda_launch(ctx->f_fill_color, _cuda_grid(npix), args);
}

static int
_cuda_copy(GPUBackend self, ImagingGPU out, ImagingGPU in) {
    return _cuda_buffer_copy(self, &out->buffer, &in->buffer, in->buffer.size);
}

static int
_cuda_histogram(GPUBackend self, ImagingGPU im, long *hist_out) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    int npix = im->xsize * im->ysize;
    int bands = im->bands, ps = im->pixelsize;
    int hist_size = bands * 256;
    memset(hist_out, 0, hist_size * sizeof(long));

    CUdeviceptr hist_d;
    cuMemAlloc(&hist_d, hist_size * sizeof(int));
    cuMemsetD32(hist_d, 0, hist_size);

    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    void *args[] = {&pi, &hist_d, &npix, &ps, &bands};
    int err = _cuda_launch(ctx->f_histogram, _cuda_grid(npix), args);

    int *tmp = (int *)malloc(hist_size * sizeof(int));
    cuMemcpyDtoH(tmp, hist_d, hist_size * sizeof(int));
    for (int i = 0; i < hist_size; i++) hist_out[i] = tmp[i];
    free(tmp);
    cuMemFree(hist_d);
    return err;
}

static int
_cuda_transform(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    int method,
    double a[8],
    int filter,
    int fill
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    int iw = in->xsize, ih = in->ysize, ow = out->xsize, oh = out->ysize;
    int ps = in->pixelsize;
    size_t total = (size_t)ow * oh;

    if (method == 2) { /* PERSPECTIVE */
        float fa[8];
        for (int i = 0; i < 8; i++) fa[i] = (float)a[i];
        void *args[] = {
            &pi,
            &po,
            &iw,
            &ih,
            &ow,
            &oh,
            &ps,
            &fa[0],
            &fa[1],
            &fa[2],
            &fa[3],
            &fa[4],
            &fa[5],
            &fa[6],
            &fa[7],
            &filter,
            &fill
        };
        return _cuda_launch(ctx->f_perspective_transform, _cuda_grid(total), args);
    } else { /* AFFINE */
        float fa[6];
        for (int i = 0; i < 6; i++) fa[i] = (float)a[i];
        void *args[] = {
            &pi,
            &po,
            &iw,
            &ih,
            &ow,
            &oh,
            &ps,
            &fa[0],
            &fa[1],
            &fa[2],
            &fa[3],
            &fa[4],
            &fa[5],
            &filter,
            &fill
        };
        return _cuda_launch(ctx->f_affine_transform, _cuda_grid(total), args);
    }
}

/* ================================================================== */
/* NEW CUDA operations                                                  */
/* ================================================================== */

static int
_cuda_paste(
    GPUBackend self,
    ImagingGPU dest,
    ImagingGPU src,
    ImagingGPU mask,
    int dx,
    int dy,
    int sx,
    int sy,
    int sw,
    int sh
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_paste_with_mask)
        return GPU_ERROR_UNSUPPORTED;

    int npix = sw * sh;
    if (npix <= 0)
        return GPU_OK;

    CUdeviceptr pd = (CUdeviceptr)dest->buffer.handle.cu_ptr;
    CUdeviceptr ps = (CUdeviceptr)src->buffer.handle.cu_ptr;
    CUdeviceptr pm;
    int free_mask = 0;

    if (mask && mask->buffer.handle.cu_ptr) {
        pm = (CUdeviceptr)mask->buffer.handle.cu_ptr;
    } else {
        cuMemAlloc(&pm, npix);
        cuMemsetD8(pm, 255, npix);
        free_mask = 1;
    }

    int dw = dest->xsize, dh = dest->ysize;
    int pxs = dest->pixelsize;
    void *args[] = {&pd, &ps, &pm, &dw, &dh, &sw, &sh, &pxs, &dx, &dy};
    int err = _cuda_launch(ctx->f_paste_with_mask, _cuda_grid(npix), args);

    if (free_mask)
        cuMemFree(pm);
    return err;
}

static int
_cuda_merge(GPUBackend self, ImagingGPU out, ImagingGPU bands[4], int nbands) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_merge_bands)
        return GPU_ERROR_UNSUPPORTED;

    int npix = out->xsize * out->ysize;
    CUdeviceptr b0 = (CUdeviceptr)bands[0]->buffer.handle.cu_ptr;
    CUdeviceptr b1 = (nbands >= 2) ? (CUdeviceptr)bands[1]->buffer.handle.cu_ptr : b0;
    CUdeviceptr b2 = (nbands >= 3) ? (CUdeviceptr)bands[2]->buffer.handle.cu_ptr : b0;
    CUdeviceptr b3 = (nbands >= 4) ? (CUdeviceptr)bands[3]->buffer.handle.cu_ptr : b0;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;

    void *args[] = {&b0, &b1, &b2, &b3, &po, &npix, &nbands};
    return _cuda_launch(ctx->f_merge_bands, _cuda_grid(npix), args);
}

static int
_cuda_split(GPUBackend self, ImagingGPU im, ImagingGPU bands[4]) {
    for (int i = 0; i < im->bands; i++) {
        int err = _cuda_getband(self, bands[i], im, i);
        if (err != GPU_OK)
            return err;
    }
    return GPU_OK;
}

static int
_cuda_getbbox(GPUBackend self, ImagingGPU im, int bbox[4], int alpha_only) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_getbbox) {
        /* Fallback: not available */
        return GPU_ERROR_UNSUPPORTED;
    }

    /* Allocate result buffer on GPU: [min_x, min_y, max_x, max_y] */
    int init_vals[4] = {im->xsize, im->ysize, -1, -1};
    CUdeviceptr d_result;
    cuMemAlloc(&d_result, 4 * sizeof(int));
    cuMemcpyHtoD(d_result, init_vals, 4 * sizeof(int));

    int npix = im->xsize * im->ysize;
    int check_ch = alpha_only ? (im->pixelsize - 1) : -1;
    int w = im->xsize, h = im->ysize, ps = im->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    void *args[] = {&pi, &d_result, &w, &h, &ps, &check_ch};
    int err = _cuda_launch(ctx->f_getbbox, _cuda_grid(npix), args);
    if (err != GPU_OK) {
        cuMemFree(d_result);
        return err;
    }

    int results[4];
    cuMemcpyDtoH(results, d_result, 4 * sizeof(int));
    cuMemFree(d_result);

    if (results[2] < results[0]) {
        bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0;
    } else {
        bbox[0] = results[0];
        bbox[1] = results[1];
        bbox[2] = results[2] + 1;
        bbox[3] = results[3] + 1;
    }
    return GPU_OK;
}

static int
_cuda_getextrema(GPUBackend self, ImagingGPU im, void *extrema) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_getextrema)
        return GPU_ERROR_UNSUPPORTED;

    int bands = im->bands;
    int init_vals[8];
    for (int b = 0; b < bands; b++) {
        init_vals[b * 2] = 255;
        init_vals[b * 2 + 1] = 0;
    }

    CUdeviceptr d_result;
    cuMemAlloc(&d_result, bands * 2 * sizeof(int));
    cuMemcpyHtoD(d_result, init_vals, bands * 2 * sizeof(int));

    int npix = im->xsize * im->ysize;
    int ps = im->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)im->buffer.handle.cu_ptr;
    void *args[] = {&pi, &d_result, &npix, &ps, &bands};
    int err = _cuda_launch(ctx->f_getextrema, _cuda_grid(npix), args);
    if (err != GPU_OK) {
        cuMemFree(d_result);
        return err;
    }

    int results[8];
    cuMemcpyDtoH(results, d_result, bands * 2 * sizeof(int));
    cuMemFree(d_result);

    UINT8 *ext = (UINT8 *)extrema;
    for (int b = 0; b < bands; b++) {
        ext[b * 2] = (UINT8)results[b * 2];
        ext[b * 2 + 1] = (UINT8)results[b * 2 + 1];
    }
    return GPU_OK;
}

static int
_cuda_effect_spread(GPUBackend self, ImagingGPU out, ImagingGPU in, int distance) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_effect_spread)
        return GPU_ERROR_UNSUPPORTED;

    int npix = in->xsize * in->ysize;
    int w = in->xsize, h = in->ysize, ps = in->pixelsize;
    unsigned int seed = (unsigned int)(npix * 7919 + 12345);
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &w, &h, &ps, &distance, &seed};
    return _cuda_launch(ctx->f_effect_spread, _cuda_grid(npix), args);
}

static int
_cuda_color_matrix(
    GPUBackend self, ImagingGPU out, ImagingGPU in, const float *matrix, int ncolumns
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_color_matrix)
        return GPU_ERROR_UNSUPPORTED;

    int npix = in->xsize * in->ysize;
    int pxs = in->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;

    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int r = 0; r < 4 && r < ncolumns; r++)
        for (int c = 0; c < 4 && c < ncolumns; c++)
            m[r * 4 + c] = matrix[r * ncolumns + c];

    void *args[] = {&pi,    &po,    &npix,  &pxs,   &m[0],  &m[1], &m[2],
                    &m[3],  &m[4],  &m[5],  &m[6],  &m[7],  &m[8], &m[9],
                    &m[10], &m[11], &m[12], &m[13], &m[14], &m[15]};
    return _cuda_launch(ctx->f_color_matrix, _cuda_grid(npix), args);
}

static int
_cuda_color_lut_3d(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    const INT16 *table
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_color_lut_3d_gpu) {
        /* Fallback: not available */
        return GPU_ERROR_UNSUPPORTED;
    }

    int npix = in->xsize * in->ysize;
    int ps = in->pixelsize;

    /* Upload LUT table to GPU */
    size_t table_size =
        (size_t)size1D * size2D * size3D * table_channels * sizeof(short);
    CUdeviceptr d_table;
    CUresult cerr = cuMemAlloc(&d_table, table_size);
    if (cerr != CUDA_SUCCESS)
        return GPU_ERROR_MEMORY;
    cuMemcpyHtoD(d_table, table, table_size);

    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {
        &pi, &po, &d_table, &npix, &ps, &table_channels, &size1D, &size2D, &size3D
    };
    int err = _cuda_launch(ctx->f_color_lut_3d_gpu, _cuda_grid(npix), args);
    cuMemFree(d_table);
    return err;
}

static int
_cuda_crop(
    GPUBackend self, ImagingGPU out, ImagingGPU in, int x0, int y0, int x1, int y1
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_crop_region)
        return GPU_ERROR_UNSUPPORTED;

    int ow = x1 - x0, oh = y1 - y0, npix = ow * oh;
    int pxs = in->pixelsize, iw = in->xsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &iw, &ow, &oh, &pxs, &x0, &y0};
    return _cuda_launch(ctx->f_crop_region, _cuda_grid(npix), args);
}

static int
_cuda_expand(
    GPUBackend self,
    ImagingGPU out,
    ImagingGPU in,
    int xmargin,
    int ymargin,
    const UINT8 *fill
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_expand_image)
        return GPU_ERROR_UNSUPPORTED;

    int iw = in->xsize, ih = in->ysize;
    int ow = out->xsize, oh = out->ysize;
    int pxs = in->pixelsize;
    int npix = ow * oh;
    int f0 = fill ? fill[0] : 0;
    int f1 = fill ? (pxs >= 2 ? fill[1] : 0) : 0;
    int f2 = fill ? (pxs >= 3 ? fill[2] : 0) : 0;
    int f3 = fill ? (pxs >= 4 ? fill[3] : 0) : 0;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {
        &pi, &po, &iw, &ih, &ow, &oh, &pxs, &xmargin, &ymargin, &f0, &f1, &f2, &f3
    };
    return _cuda_launch(ctx->f_expand_image, _cuda_grid(npix), args);
}

static int
_cuda_offset(GPUBackend self, ImagingGPU out, ImagingGPU in, int xoffset, int yoffset) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_offset_image)
        return GPU_ERROR_UNSUPPORTED;

    int w = in->xsize, h = in->ysize, pxs = in->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &w, &h, &pxs, &xoffset, &yoffset};
    return _cuda_launch(ctx->f_offset_image, _cuda_grid(w * h), args);
}

static int
_cuda_negative(GPUBackend self, ImagingGPU out, ImagingGPU in) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_negative)
        return GPU_ERROR_UNSUPPORTED;

    int total = in->xsize * in->ysize * in->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &total};
    return _cuda_launch(ctx->f_negative, _cuda_grid(total), args);
}

static int
_cuda_posterize(GPUBackend self, ImagingGPU out, ImagingGPU in, int bits) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_posterize)
        return GPU_ERROR_UNSUPPORTED;

    int total = in->xsize * in->ysize * in->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &total, &bits};
    return _cuda_launch(ctx->f_posterize, _cuda_grid(total), args);
}

static int
_cuda_solarize(GPUBackend self, ImagingGPU out, ImagingGPU in, int threshold) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_solarize)
        return GPU_ERROR_UNSUPPORTED;

    int total = in->xsize * in->ysize * in->pixelsize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &total, &threshold};
    return _cuda_launch(ctx->f_solarize, _cuda_grid(total), args);
}

static int
_cuda_equalize(GPUBackend self, ImagingGPU out, ImagingGPU in, const UINT8 *lut) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_equalize)
        return GPU_ERROR_UNSUPPORTED;

    int npix = in->xsize * in->ysize;
    int pxs = in->pixelsize, bands = in->bands;
    int lut_size = bands * 256;

    CUdeviceptr lut_d;
    cuMemAlloc(&lut_d, lut_size);
    cuMemcpyHtoD(lut_d, lut, lut_size);

    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &lut_d, &npix, &pxs, &bands};
    int err = _cuda_launch(ctx->f_equalize, _cuda_grid(npix), args);
    cuMemFree(lut_d);
    return err;
}

static int
_cuda_linear_gradient(GPUBackend self, ImagingGPU out, int direction) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_linear_gradient)
        return GPU_ERROR_UNSUPPORTED;

    int w = out->xsize, h = out->ysize, npix = w * h;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&po, &w, &h, &direction};
    return _cuda_launch(ctx->f_linear_gradient, _cuda_grid(npix), args);
}

static int
_cuda_radial_gradient(GPUBackend self, ImagingGPU out) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_radial_gradient)
        return GPU_ERROR_UNSUPPORTED;

    int w = out->xsize, h = out->ysize, npix = w * h;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&po, &w, &h};
    return _cuda_launch(ctx->f_radial_gradient, _cuda_grid(npix), args);
}

static int
_cuda_reduce(
    GPUBackend self, ImagingGPU out, ImagingGPU in, int factor_x, int factor_y
) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_reduce)
        return GPU_ERROR_UNSUPPORTED;

    int out_npix = out->xsize * out->ysize;
    int in_w = in->xsize, in_h = in->ysize;
    int out_w = out->xsize, out_h = out->ysize;
    int ps = in->pixelsize, bands = in->bands;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {
        &pi, &po, &in_w, &in_h, &out_w, &out_h, &factor_x, &factor_y, &ps, &bands
    };
    return _cuda_launch(ctx->f_reduce, _cuda_grid(out_npix), args);
}

static int
_cuda_rank_filter(GPUBackend self, ImagingGPU out, ImagingGPU in, int ksize, int rank) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_rank_filter)
        return GPU_ERROR_UNSUPPORTED;
    if (ksize > 7)
        return GPU_ERROR_UNSUPPORTED;

    int npix = in->xsize * in->ysize;
    int w = in->xsize, h = in->ysize, ps = in->pixelsize, bands = in->bands;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &w, &h, &ps, &bands, &ksize, &rank};
    return _cuda_launch(ctx->f_rank_filter, _cuda_grid(npix), args);
}

static int
_cuda_mode_filter(GPUBackend self, ImagingGPU out, ImagingGPU in, int ksize) {
    CUDAContext *ctx = (CUDAContext *)self->ctx;
    if (!ctx->f_mode_filter)
        return GPU_ERROR_UNSUPPORTED;
    if (in->bands != 1)
        return GPU_ERROR_UNSUPPORTED;

    int npix = in->xsize * in->ysize;
    int w = in->xsize, h = in->ysize;
    CUdeviceptr pi = (CUdeviceptr)in->buffer.handle.cu_ptr;
    CUdeviceptr po = (CUdeviceptr)out->buffer.handle.cu_ptr;
    void *args[] = {&pi, &po, &w, &h, &ksize};
    return _cuda_launch(ctx->f_mode_filter, _cuda_grid(npix), args);
}

/* ================================================================== */
/* Backend initialization                                               */
/* ================================================================== */

GPUBackend
ImagingGPU_CUDA_Init(void) {
    CUresult err;

    err = cuInit(0);
    if (err != CUDA_SUCCESS)
        return NULL;

    CUdevice device;
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS)
        return NULL;

    CUcontext cu_ctx;
    err = cuCtxCreate(&cu_ctx, 0, device);
    if (err != CUDA_SUCCESS)
        return NULL;

    /* Compile kernel source with NVRTC */
    nvrtcProgram prog;
    nvrtcResult nvrtc_err =
        nvrtcCreateProgram(&prog, CUDA_KERNEL_SOURCE, "pillow_gpu.cu", 0, NULL, NULL);
    if (nvrtc_err != NVRTC_SUCCESS) {
        cuCtxDestroy(cu_ctx);
        return NULL;
    }

    const char *opts[] = {"--use_fast_math"};
    nvrtc_err = nvrtcCompileProgram(prog, 1, opts);
    if (nvrtc_err != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "Pillow GPU CUDA compile log:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        cuCtxDestroy(cu_ctx);
        return NULL;
    }

    /* Get PTX */
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char *ptx = (char *)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    /* Load module from PTX */
    CUmodule cu_module;
    err = cuModuleLoadDataEx(&cu_module, ptx, 0, NULL, NULL);
    free(ptx);
    if (err != CUDA_SUCCESS) {
        cuCtxDestroy(cu_ctx);
        return NULL;
    }

    /* Allocate context */
    CUDAContext *ctx = (CUDAContext *)calloc(1, sizeof(CUDAContext));
    if (!ctx) {
        cuModuleUnload(cu_module);
        cuCtxDestroy(cu_ctx);
        return NULL;
    }
    ctx->cu_context = cu_ctx;
    ctx->cu_device = device;
    ctx->cu_module = cu_module;

    /* Get all kernel functions */
    ctx->f_box_blur_h = _cuda_get_func(cu_module, "box_blur_h");
    ctx->f_box_blur_v = _cuda_get_func(cu_module, "box_blur_v");
    ctx->f_gaussian_blur_h = _cuda_get_func(cu_module, "gaussian_blur_h");
    ctx->f_gaussian_blur_v = _cuda_get_func(cu_module, "gaussian_blur_v");
    ctx->f_convolve = _cuda_get_func(cu_module, "convolve");
    ctx->f_resample_nearest = _cuda_get_func(cu_module, "resample_nearest");
    ctx->f_resample_bilinear = _cuda_get_func(cu_module, "resample_bilinear");
    ctx->f_resample_bicubic = _cuda_get_func(cu_module, "resample_bicubic");
    ctx->f_resample_lanczos = _cuda_get_func(cu_module, "resample_lanczos");
    ctx->f_convert_rgb_to_l = _cuda_get_func(cu_module, "convert_rgb_to_l");
    ctx->f_convert_l_to_rgb = _cuda_get_func(cu_module, "convert_l_to_rgb");
    ctx->f_convert_rgb_to_rgba = _cuda_get_func(cu_module, "convert_rgb_to_rgba");
    ctx->f_convert_rgba_to_rgb = _cuda_get_func(cu_module, "convert_rgba_to_rgb");
    ctx->f_convert_rgba_to_l = _cuda_get_func(cu_module, "convert_rgba_to_l");
    ctx->f_convert_l_to_rgba = _cuda_get_func(cu_module, "convert_l_to_rgba");
    ctx->f_blend = _cuda_get_func(cu_module, "blend");
    ctx->f_alpha_composite = _cuda_get_func(cu_module, "alpha_composite");
    ctx->f_chop_operation = _cuda_get_func(cu_module, "chop_operation");
    ctx->f_transpose_op = _cuda_get_func(cu_module, "transpose_op");
    ctx->f_point_lut = _cuda_get_func(cu_module, "point_lut");
    ctx->f_point_transform = _cuda_get_func(cu_module, "point_transform");
    ctx->f_getband = _cuda_get_func(cu_module, "getband");
    ctx->f_putband = _cuda_get_func(cu_module, "putband");
    ctx->f_fillband = _cuda_get_func(cu_module, "fillband");
    ctx->f_fill_color = _cuda_get_func(cu_module, "fill_color");
    ctx->f_unsharp_mask = _cuda_get_func(cu_module, "unsharp_mask");
    ctx->f_histogram = _cuda_get_func(cu_module, "histogram");
    ctx->f_affine_transform = _cuda_get_func(cu_module, "affine_transform");
    ctx->f_perspective_transform = _cuda_get_func(cu_module, "perspective_transform");
    ctx->f_convert_rgb_to_cmyk = _cuda_get_func(cu_module, "convert_rgb_to_cmyk");
    ctx->f_convert_cmyk_to_rgb = _cuda_get_func(cu_module, "convert_cmyk_to_rgb");
    ctx->f_merge_bands = _cuda_get_func(cu_module, "merge_bands");
    ctx->f_paste_with_mask = _cuda_get_func(cu_module, "paste_with_mask");
    ctx->f_color_matrix = _cuda_get_func(cu_module, "color_matrix");
    ctx->f_crop_region = _cuda_get_func(cu_module, "crop_region");
    ctx->f_expand_image = _cuda_get_func(cu_module, "expand_image");
    ctx->f_offset_image = _cuda_get_func(cu_module, "offset_image");
    ctx->f_negative = _cuda_get_func(cu_module, "negative");
    ctx->f_posterize = _cuda_get_func(cu_module, "posterize");
    ctx->f_solarize = _cuda_get_func(cu_module, "solarize");
    ctx->f_equalize = _cuda_get_func(cu_module, "equalize");
    ctx->f_linear_gradient = _cuda_get_func(cu_module, "linear_gradient");
    ctx->f_radial_gradient = _cuda_get_func(cu_module, "radial_gradient");
    ctx->f_getbbox = _cuda_get_func(cu_module, "getbbox");
    ctx->f_getextrema = _cuda_get_func(cu_module, "getextrema");
    ctx->f_effect_spread = _cuda_get_func(cu_module, "effect_spread");
    ctx->f_convert_ycbcr_to_rgb = _cuda_get_func(cu_module, "convert_ycbcr_to_rgb");
    ctx->f_convert_rgb_to_ycbcr = _cuda_get_func(cu_module, "convert_rgb_to_ycbcr");
    ctx->f_reduce = _cuda_get_func(cu_module, "reduce_image");
    ctx->f_rank_filter = _cuda_get_func(cu_module, "rank_filter");
    ctx->f_mode_filter = _cuda_get_func(cu_module, "mode_filter");
    ctx->f_color_lut_3d_gpu = _cuda_get_func(cu_module, "color_lut_3d");

    /* Allocate backend */
    GPUBackend backend = (GPUBackend)calloc(1, sizeof(struct GPUBackendInstance));
    if (!backend) {
        free(ctx);
        cuModuleUnload(cu_module);
        cuCtxDestroy(cu_ctx);
        return NULL;
    }

    backend->type = GPU_BACKEND_CUDA;
    backend->name = "CUDA";
    backend->ctx = ctx;

    /* Query device info */
    cuDeviceGetName(backend->device_name, sizeof(backend->device_name), device);
    size_t total_mem;
    cuDeviceTotalMem(&total_mem, device);
    backend->total_mem = total_mem;
    backend->max_mem_alloc = total_mem; /* CUDA doesn't have a separate limit */

    /* Assign function pointers */
    backend->shutdown = _cuda_shutdown;
    backend->buffer_alloc = _cuda_buffer_alloc;
    backend->buffer_free = _cuda_buffer_free;
    backend->buffer_upload = _cuda_buffer_upload;
    backend->buffer_download = _cuda_buffer_download;
    backend->buffer_copy = _cuda_buffer_copy;
    backend->gaussian_blur = _cuda_gaussian_blur;
    backend->box_blur = _cuda_box_blur;
    backend->unsharp_mask = _cuda_unsharp_mask;
    backend->filter = _cuda_filter;
    backend->resample = _cuda_resample;
    backend->convert = _cuda_convert;
    backend->blend = _cuda_blend;
    backend->alpha_composite = _cuda_alpha_composite;
    backend->chop = _cuda_chop;
    backend->transpose = _cuda_transpose;
    backend->transform = _cuda_transform;
    backend->point_lut = _cuda_point_lut;
    backend->point_transform = _cuda_point_transform;
    backend->fill = _cuda_fill;
    backend->copy = _cuda_copy;
    backend->getband = _cuda_getband;
    backend->putband = _cuda_putband;
    backend->fillband = _cuda_fillband;
    backend->histogram = _cuda_histogram;

    /* All operations now implemented */
    backend->color_matrix = _cuda_color_matrix;
    backend->color_lut_3d = _cuda_color_lut_3d;
    backend->paste = _cuda_paste;
    backend->getbbox = _cuda_getbbox;
    backend->getextrema = _cuda_getextrema;
    backend->merge = _cuda_merge;
    backend->split = _cuda_split;
    backend->effect_spread = _cuda_effect_spread;
    backend->crop = _cuda_crop;
    backend->expand = _cuda_expand;
    backend->offset = _cuda_offset;
    backend->negative = _cuda_negative;
    backend->posterize = _cuda_posterize;
    backend->solarize = _cuda_solarize;
    backend->equalize = _cuda_equalize;
    backend->linear_gradient = _cuda_linear_gradient;
    backend->radial_gradient = _cuda_radial_gradient;
    backend->reduce = _cuda_reduce;
    backend->rank_filter = _cuda_rank_filter;
    backend->mode_filter = _cuda_mode_filter;

    return backend;
}

#endif /* HAVE_CUDA */
