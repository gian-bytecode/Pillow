/*
 * Pillow GPU Acceleration Layer
 *
 * GPU image lifecycle, backend initialization, CPU↔GPU transfer,
 * and operation dispatch to the active backend.
 *
 * Copyright (c) 2026 Pillow Contributors
 */

#include "GpuImaging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------- */
/* Global backend singleton                                              */
/* -------------------------------------------------------------------- */

static GPUBackend _active_backend = NULL;
static int _backend_init_attempted = 0;

/* -------------------------------------------------------------------- */
/* Mode utilities                                                        */
/* -------------------------------------------------------------------- */

int
ImagingGPU_ModeToPixelSize(ModeID mode) {
    switch (mode) {
        case IMAGING_MODE_1:
        case IMAGING_MODE_L:
        case IMAGING_MODE_P:
            return 1;
        case IMAGING_MODE_I_16:
        case IMAGING_MODE_I_16L:
        case IMAGING_MODE_I_16B:
        case IMAGING_MODE_I_16N:
            return 2;
        case IMAGING_MODE_I:
        case IMAGING_MODE_F:
        case IMAGING_MODE_RGB:
        case IMAGING_MODE_RGBA:
        case IMAGING_MODE_RGBX:
        case IMAGING_MODE_RGBa:
        case IMAGING_MODE_CMYK:
        case IMAGING_MODE_YCbCr:
        case IMAGING_MODE_LAB:
        case IMAGING_MODE_HSV:
        case IMAGING_MODE_LA:
        case IMAGING_MODE_La:
        case IMAGING_MODE_PA:
            return 4;
        default:
            return 0;
    }
}

int
ImagingGPU_ModeToType(ModeID mode) {
    switch (mode) {
        case IMAGING_MODE_I:
            return IMAGING_TYPE_INT32;
        case IMAGING_MODE_F:
            return IMAGING_TYPE_FLOAT32;
        case IMAGING_MODE_I_16:
        case IMAGING_MODE_I_16L:
        case IMAGING_MODE_I_16B:
        case IMAGING_MODE_I_16N:
            return IMAGING_TYPE_SPECIAL;
        default:
            return IMAGING_TYPE_UINT8;
    }
}

int
ImagingGPU_ModeToBands(ModeID mode) {
    switch (mode) {
        case IMAGING_MODE_1:
        case IMAGING_MODE_L:
        case IMAGING_MODE_P:
        case IMAGING_MODE_I:
        case IMAGING_MODE_F:
        case IMAGING_MODE_I_16:
        case IMAGING_MODE_I_16L:
        case IMAGING_MODE_I_16B:
        case IMAGING_MODE_I_16N:
            return 1;
        case IMAGING_MODE_LA:
        case IMAGING_MODE_La:
        case IMAGING_MODE_PA:
            return 2;
        case IMAGING_MODE_RGB:
        case IMAGING_MODE_RGBX:
        case IMAGING_MODE_YCbCr:
        case IMAGING_MODE_LAB:
        case IMAGING_MODE_HSV:
            return 3;
        case IMAGING_MODE_RGBA:
        case IMAGING_MODE_RGBa:
        case IMAGING_MODE_CMYK:
            return 4;
        default:
            return 0;
    }
}

/* -------------------------------------------------------------------- */
/* Backend initialization                                                */
/* -------------------------------------------------------------------- */

GPUBackend
ImagingGPU_BackendInit(int preferred_type) {
    if (_active_backend) {
        return _active_backend;
    }
    _backend_init_attempted = 1;

    /* Try preferred backend first */
    if (preferred_type == GPU_BACKEND_CUDA) {
#ifdef HAVE_CUDA
        _active_backend = ImagingGPU_CUDA_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
#ifdef HAVE_OPENCL
        _active_backend = ImagingGPU_OpenCL_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
    } else if (preferred_type == GPU_BACKEND_OPENCL) {
#ifdef HAVE_OPENCL
        _active_backend = ImagingGPU_OpenCL_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
#ifdef HAVE_CUDA
        _active_backend = ImagingGPU_CUDA_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
    } else {
        /* Auto: try CUDA first (generally better perf), fall back to OpenCL */
#ifdef HAVE_CUDA
        _active_backend = ImagingGPU_CUDA_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
#ifdef HAVE_OPENCL
        _active_backend = ImagingGPU_OpenCL_Init();
        if (_active_backend) {
            return _active_backend;
        }
#endif
    }

    return NULL;
}

GPUBackend
ImagingGPU_GetBackend(void) {
    if (!_active_backend && !_backend_init_attempted) {
        ImagingGPU_BackendInit(0);
    }
    return _active_backend;
}

void
ImagingGPU_BackendShutdown(void) {
    if (_active_backend) {
        if (_active_backend->shutdown) {
            _active_backend->shutdown(_active_backend);
        }
        _active_backend = NULL;
    }
    _backend_init_attempted = 0;
}

int
ImagingGPU_IsAvailable(void) {
    return ImagingGPU_GetBackend() != NULL;
}

const char *
ImagingGPU_GetBackendName(void) {
    GPUBackend b = ImagingGPU_GetBackend();
    return b ? b->name : "none";
}

const char *
ImagingGPU_GetDeviceName(void) {
    GPUBackend b = ImagingGPU_GetBackend();
    return b ? b->device_name : "";
}

/* -------------------------------------------------------------------- */
/* GPU Image lifecycle                                                   */
/* -------------------------------------------------------------------- */

static ImagingGPU
_gpu_image_new(ModeID mode, int xsize, int ysize, int dirty) {
    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return (ImagingGPU)ImagingError_ValueError("No GPU backend available");
    }

    int pixelsize = ImagingGPU_ModeToPixelSize(mode);
    if (pixelsize == 0) {
        return (ImagingGPU)ImagingError_ModeError();
    }
    if (xsize <= 0 || ysize <= 0) {
        return (ImagingGPU)ImagingError_ValueError("Invalid image dimensions");
    }

    ImagingGPU im = (ImagingGPU)calloc(1, sizeof(struct ImagingGPUInstance));
    if (!im) {
        return (ImagingGPU)ImagingError_MemoryError();
    }

    im->mode = mode;
    im->type = ImagingGPU_ModeToType(mode);
    im->bands = ImagingGPU_ModeToBands(mode);
    im->xsize = xsize;
    im->ysize = ysize;
    im->pixelsize = pixelsize;
    im->linesize = xsize * pixelsize;
    im->backend_type = backend->type;
    im->palette = NULL;

    size_t total_size = (size_t)im->linesize * (size_t)ysize;
    int err = backend->buffer_alloc(backend, &im->buffer, total_size);
    if (err != GPU_OK) {
        free(im);
        return (ImagingGPU)ImagingError_MemoryError();
    }

    /* Zero the buffer unless dirty */
    if (!dirty) {
        void *zeros = calloc(1, total_size);
        if (zeros) {
            backend->buffer_upload(backend, &im->buffer, zeros, total_size);
            free(zeros);
        }
    }

    return im;
}

ImagingGPU
ImagingGPU_New(ModeID mode, int xsize, int ysize) {
    return _gpu_image_new(mode, xsize, ysize, 0);
}

ImagingGPU
ImagingGPU_NewDirty(ModeID mode, int xsize, int ysize) {
    return _gpu_image_new(mode, xsize, ysize, 1);
}

void
ImagingGPU_Delete(ImagingGPU im) {
    if (!im) {
        return;
    }
    GPUBackend backend = ImagingGPU_GetBackend();
    if (backend && im->buffer.size > 0) {
        backend->buffer_free(backend, &im->buffer);
    }
    if (im->palette) {
        ImagingPaletteDelete(im->palette);
    }
    free(im);
}

/* -------------------------------------------------------------------- */
/* CPU ↔ GPU transfer                                                    */
/* -------------------------------------------------------------------- */

ImagingGPU
ImagingGPU_FromImaging(Imaging im) {
    if (!im) {
        return (ImagingGPU)ImagingError_ValueError("NULL image");
    }

    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return (ImagingGPU)ImagingError_ValueError("No GPU backend available");
    }

    ImagingGPU gpu_im = ImagingGPU_NewDirty(im->mode, im->xsize, im->ysize);
    if (!gpu_im) {
        return NULL;
    }

    /* Copy palette if present */
    if (im->palette) {
        gpu_im->palette = ImagingPaletteDuplicate(im->palette);
    }

    /* Gather rows into contiguous buffer, then upload */
    size_t total_size = (size_t)im->linesize * (size_t)im->ysize;
    char *host_buf = (char *)malloc(total_size);
    if (!host_buf) {
        ImagingGPU_Delete(gpu_im);
        return (ImagingGPU)ImagingError_MemoryError();
    }

    for (int y = 0; y < im->ysize; y++) {
        memcpy(host_buf + (size_t)y * im->linesize, im->image[y], im->linesize);
    }

    int err = backend->buffer_upload(backend, &gpu_im->buffer, host_buf, total_size);
    free(host_buf);

    if (err != GPU_OK) {
        ImagingGPU_Delete(gpu_im);
        return (ImagingGPU)ImagingError_ValueError("GPU upload failed");
    }

    return gpu_im;
}

ImagingGPU
ImagingGPU_FromBytes(
    ModeID mode, int xsize, int ysize, const void *data, size_t data_len
) {
    if (!data) {
        return (ImagingGPU)ImagingError_ValueError("NULL data");
    }
    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return (ImagingGPU)ImagingError_ValueError("No GPU backend available");
    }

    ImagingGPU gpu_im = ImagingGPU_NewDirty(mode, xsize, ysize);
    if (!gpu_im) {
        return NULL;
    }

    size_t expected = (size_t)gpu_im->linesize * (size_t)ysize;
    if (data_len < expected) {
        ImagingGPU_Delete(gpu_im);
        return (ImagingGPU)ImagingError_ValueError("data buffer too small");
    }

    int err = backend->buffer_upload(backend, &gpu_im->buffer, data, expected);
    if (err != GPU_OK) {
        ImagingGPU_Delete(gpu_im);
        return (ImagingGPU)ImagingError_ValueError("GPU upload failed");
    }
    return gpu_im;
}

int
ImagingGPU_DownloadRaw(ImagingGPU gpu_im, void *out_buf, size_t buf_size) {
    if (!gpu_im || !out_buf) {
        return GPU_ERROR_INVALID_ARG;
    }
    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return GPU_ERROR_NO_BACKEND;
    }
    size_t total_size = (size_t)gpu_im->linesize * (size_t)gpu_im->ysize;
    if (buf_size < total_size) {
        return GPU_ERROR_INVALID_ARG;
    }
    return backend->buffer_download(backend, &gpu_im->buffer, out_buf, total_size);
}

Imaging
ImagingGPU_ToImaging(ImagingGPU gpu_im) {
    if (!gpu_im) {
        return (Imaging)ImagingError_ValueError("NULL GPU image");
    }

    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return (Imaging)ImagingError_ValueError("No GPU backend available");
    }

    Imaging im = ImagingNew(gpu_im->mode, gpu_im->xsize, gpu_im->ysize);
    if (!im) {
        return NULL;
    }

    /* Copy palette */
    if (gpu_im->palette) {
        im->palette = ImagingPaletteDuplicate(gpu_im->palette);
    }

    /* Download to contiguous buffer, then scatter to rows */
    size_t total_size = (size_t)gpu_im->linesize * (size_t)gpu_im->ysize;
    char *host_buf = (char *)malloc(total_size);
    if (!host_buf) {
        ImagingDelete(im);
        return (Imaging)ImagingError_MemoryError();
    }

    int err = backend->buffer_download(backend, &gpu_im->buffer, host_buf, total_size);
    if (err != GPU_OK) {
        free(host_buf);
        ImagingDelete(im);
        return (Imaging)ImagingError_ValueError("GPU download failed");
    }

    for (int y = 0; y < gpu_im->ysize; y++) {
        memcpy(im->image[y], host_buf + (size_t)y * gpu_im->linesize, gpu_im->linesize);
    }
    free(host_buf);

    return im;
}

int
ImagingGPU_Upload(ImagingGPU gpu_im, Imaging im) {
    if (!gpu_im || !im) {
        return GPU_ERROR_INVALID_ARG;
    }
    if (gpu_im->mode != im->mode || gpu_im->xsize != im->xsize ||
        gpu_im->ysize != im->ysize) {
        return GPU_ERROR_INVALID_ARG;
    }

    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return GPU_ERROR_NO_BACKEND;
    }

    size_t total_size = (size_t)im->linesize * (size_t)im->ysize;
    char *host_buf = (char *)malloc(total_size);
    if (!host_buf) {
        return GPU_ERROR_MEMORY;
    }

    for (int y = 0; y < im->ysize; y++) {
        memcpy(host_buf + (size_t)y * im->linesize, im->image[y], im->linesize);
    }

    int err = backend->buffer_upload(backend, &gpu_im->buffer, host_buf, total_size);
    free(host_buf);
    return err;
}

int
ImagingGPU_Download(ImagingGPU gpu_im, Imaging im) {
    if (!gpu_im || !im) {
        return GPU_ERROR_INVALID_ARG;
    }
    if (gpu_im->mode != im->mode || gpu_im->xsize != im->xsize ||
        gpu_im->ysize != im->ysize) {
        return GPU_ERROR_INVALID_ARG;
    }

    GPUBackend backend = ImagingGPU_GetBackend();
    if (!backend) {
        return GPU_ERROR_NO_BACKEND;
    }

    size_t total_size = (size_t)gpu_im->linesize * (size_t)gpu_im->ysize;
    char *host_buf = (char *)malloc(total_size);
    if (!host_buf) {
        return GPU_ERROR_MEMORY;
    }

    int err = backend->buffer_download(backend, &gpu_im->buffer, host_buf, total_size);
    if (err != GPU_OK) {
        free(host_buf);
        return err;
    }

    for (int y = 0; y < gpu_im->ysize; y++) {
        memcpy(im->image[y], host_buf + (size_t)y * gpu_im->linesize, gpu_im->linesize);
    }
    free(host_buf);
    return GPU_OK;
}

/* -------------------------------------------------------------------- */
/* Operation dispatch macros                                             */
/* -------------------------------------------------------------------- */

#define GET_BACKEND_OR_FAIL()                \
    GPUBackend _b = ImagingGPU_GetBackend(); \
    if (!_b)                                 \
    return (ImagingGPU)ImagingError_ValueError("No GPU backend")

#define CHECK_OP_OR_FAIL(op_name) \
    if (!_b->op_name)             \
    return (ImagingGPU)ImagingError_ValueError("GPU backend does not support " #op_name)

/* -------------------------------------------------------------------- */
/* GPU operation wrappers (dispatch to backend)                          */
/* -------------------------------------------------------------------- */

ImagingGPU
ImagingGPU_GaussianBlur(ImagingGPU imIn, float xradius, float yradius, int passes) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(gaussian_blur);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->gaussian_blur(_b, out, imIn, xradius, yradius, passes);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU gaussian_blur failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_BoxBlur(ImagingGPU imIn, float xradius, float yradius, int n) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(box_blur);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->box_blur(_b, out, imIn, xradius, yradius, n);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU box_blur failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_UnsharpMask(ImagingGPU imIn, float radius, int percent, int threshold) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(unsharp_mask);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->unsharp_mask(_b, out, imIn, radius, percent, threshold);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU unsharp_mask failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Filter(
    ImagingGPU imIn,
    int ksize_x,
    int ksize_y,
    const float *kernel,
    float divisor,
    float offset
) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(filter);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->filter(_b, out, imIn, ksize_x, ksize_y, kernel, divisor, offset);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU filter failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Resample(ImagingGPU imIn, int xsize, int ysize, int filter, float box[4]) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(resample);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, xsize, ysize);
    if (!out)
        return NULL;

    int err = _b->resample(_b, out, imIn, xsize, ysize, filter, box);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU resample failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Convert(ImagingGPU imIn, ModeID to_mode) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(convert);

    ImagingGPU out = ImagingGPU_NewDirty(to_mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->convert(_b, out, imIn, to_mode);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU convert failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Blend(ImagingGPU im1, ImagingGPU im2, float alpha) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(blend);

    if (im1->mode != im2->mode || im1->xsize != im2->xsize ||
        im1->ysize != im2->ysize) {
        return (ImagingGPU)ImagingError_Mismatch();
    }

    ImagingGPU out = ImagingGPU_NewDirty(im1->mode, im1->xsize, im1->ysize);
    if (!out)
        return NULL;

    int err = _b->blend(_b, out, im1, im2, alpha);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU blend failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_AlphaComposite(ImagingGPU im1, ImagingGPU im2) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(alpha_composite);

    if (im1->mode != im2->mode || im1->xsize != im2->xsize ||
        im1->ysize != im2->ysize) {
        return (ImagingGPU)ImagingError_Mismatch();
    }

    ImagingGPU out = ImagingGPU_NewDirty(im1->mode, im1->xsize, im1->ysize);
    if (!out)
        return NULL;

    int err = _b->alpha_composite(_b, out, im1, im2);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU alpha_composite failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Chop(ImagingGPU im1, ImagingGPU im2, int op, float scale, int offset) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(chop);

    if (im1->xsize != im2->xsize || im1->ysize != im2->ysize) {
        return (ImagingGPU)ImagingError_Mismatch();
    }

    ImagingGPU out = ImagingGPU_NewDirty(im1->mode, im1->xsize, im1->ysize);
    if (!out)
        return NULL;

    int err = _b->chop(_b, out, im1, im2, op, scale, offset);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU chop operation failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Transpose(ImagingGPU imIn, int op) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(transpose);

    int out_w = imIn->xsize, out_h = imIn->ysize;
    /* Ops that swap dimensions */
    if (op == GPU_TRANSPOSE_ROTATE_90 || op == GPU_TRANSPOSE_ROTATE_270 ||
        op == GPU_TRANSPOSE_TRANSPOSE || op == GPU_TRANSPOSE_TRANSVERSE) {
        out_w = imIn->ysize;
        out_h = imIn->xsize;
    }

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, out_w, out_h);
    if (!out)
        return NULL;

    int err = _b->transpose(_b, out, imIn, op);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU transpose failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Transform(
    ImagingGPU imIn, int xsize, int ysize, int method, double a[8], int filter, int fill
) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(transform);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, xsize, ysize);
    if (!out)
        return NULL;

    int err = _b->transform(_b, out, imIn, method, a, filter, fill);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU transform failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_PointLut(ImagingGPU imIn, const UINT8 *lut, int bands) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(point_lut);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->point_lut(_b, out, imIn, lut, bands);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU point_lut failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_PointTransform(ImagingGPU imIn, double scale, double offset) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(point_transform);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->point_transform(_b, out, imIn, scale, offset);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU point_transform failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_ColorMatrix(
    ImagingGPU imIn, ModeID out_mode, const float *matrix, int ncolumns
) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(color_matrix);

    ImagingGPU out = ImagingGPU_NewDirty(out_mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->color_matrix(_b, out, imIn, matrix, ncolumns);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU color_matrix failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_ColorLUT3D(
    ImagingGPU imIn,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    const INT16 *table
) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(color_lut_3d);

    ModeID out_mode = table_channels == 4 ? IMAGING_MODE_RGBA : imIn->mode;
    ImagingGPU out = ImagingGPU_NewDirty(out_mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err =
        _b->color_lut_3d(_b, out, imIn, table_channels, size1D, size2D, size3D, table);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU color_lut_3d failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Fill(ModeID mode, int xsize, int ysize, const void *color) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(fill);

    ImagingGPU out = ImagingGPU_NewDirty(mode, xsize, ysize);
    if (!out)
        return NULL;

    int err = _b->fill(_b, out, color);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU fill failed");
    }
    return out;
}

int
ImagingGPU_PasteInPlace(
    ImagingGPU dest, ImagingGPU src, ImagingGPU mask, int dx, int dy
) {
    GPUBackend b = ImagingGPU_GetBackend();
    if (!b)
        return GPU_ERROR_NO_BACKEND;
    if (!b->paste)
        return GPU_ERROR_UNSUPPORTED;

    return b->paste(b, dest, src, mask, dx, dy, 0, 0, src->xsize, src->ysize);
}

ImagingGPU
ImagingGPU_Copy(ImagingGPU imIn) {
    GET_BACKEND_OR_FAIL();

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    if (_b->copy) {
        int err = _b->copy(_b, out, imIn);
        if (err != GPU_OK) {
            ImagingGPU_Delete(out);
            return (ImagingGPU)ImagingError_ValueError("GPU copy failed");
        }
    } else if (_b->buffer_copy) {
        int err = _b->buffer_copy(_b, &out->buffer, &imIn->buffer, imIn->buffer.size);
        if (err != GPU_OK) {
            ImagingGPU_Delete(out);
            return (ImagingGPU)ImagingError_ValueError("GPU copy failed");
        }
    }

    /* Copy palette */
    if (imIn->palette) {
        out->palette = ImagingPaletteDuplicate(imIn->palette);
    }

    return out;
}

ImagingGPU
ImagingGPU_GetBand(ImagingGPU imIn, int band) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(getband);

    ImagingGPU out = ImagingGPU_NewDirty(IMAGING_MODE_L, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->getband(_b, out, imIn, band);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU getband failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_PutBand(ImagingGPU imIn, ImagingGPU band_im, int band) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(putband);

    ImagingGPU out = ImagingGPU_Copy(imIn);
    if (!out)
        return NULL;

    int err = _b->putband(_b, out, band_im, band);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU putband failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_FillBand(ImagingGPU imIn, int band, int color) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(fillband);

    ImagingGPU out = ImagingGPU_Copy(imIn);
    if (!out)
        return NULL;

    int err = _b->fillband(_b, out, band, color);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU fillband failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Merge(ModeID mode, ImagingGPU bands[4]) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(merge);

    int nbands = ImagingGPU_ModeToBands(mode);
    ImagingGPU out = ImagingGPU_NewDirty(mode, bands[0]->xsize, bands[0]->ysize);
    if (!out)
        return NULL;

    int err = _b->merge(_b, out, bands, nbands);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU merge failed");
    }
    return out;
}

int
ImagingGPU_Split(ImagingGPU imIn, ImagingGPU bands[4]) {
    GPUBackend b = ImagingGPU_GetBackend();
    if (!b)
        return GPU_ERROR_NO_BACKEND;
    if (!b->split)
        return GPU_ERROR_UNSUPPORTED;

    for (int i = 0; i < imIn->bands; i++) {
        bands[i] = ImagingGPU_NewDirty(IMAGING_MODE_L, imIn->xsize, imIn->ysize);
        if (!bands[i]) {
            for (int j = 0; j < i; j++) ImagingGPU_Delete(bands[j]);
            return GPU_ERROR_MEMORY;
        }
    }

    return b->split(b, imIn, bands);
}

int
ImagingGPU_Histogram(ImagingGPU imIn, long *hist_out) {
    GPUBackend b = ImagingGPU_GetBackend();
    if (!b)
        return GPU_ERROR_NO_BACKEND;
    if (!b->histogram)
        return GPU_ERROR_UNSUPPORTED;
    return b->histogram(b, imIn, hist_out);
}

int
ImagingGPU_GetBBox(ImagingGPU imIn, int bbox[4], int alpha_only) {
    GPUBackend b = ImagingGPU_GetBackend();
    if (!b)
        return GPU_ERROR_NO_BACKEND;
    if (!b->getbbox)
        return GPU_ERROR_UNSUPPORTED;
    return b->getbbox(b, imIn, bbox, alpha_only);
}

int
ImagingGPU_GetExtrema(ImagingGPU imIn, void *extrema) {
    GPUBackend b = ImagingGPU_GetBackend();
    if (!b)
        return GPU_ERROR_NO_BACKEND;
    if (!b->getextrema)
        return GPU_ERROR_UNSUPPORTED;
    return b->getextrema(b, imIn, extrema);
}

ImagingGPU
ImagingGPU_EffectSpread(ImagingGPU imIn, int distance) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(effect_spread);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->effect_spread(_b, out, imIn, distance);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU effect_spread failed");
    }
    return out;
}

/* ================================================================== */
/* Extended operations                                                  */
/* ================================================================== */

ImagingGPU
ImagingGPU_Crop(ImagingGPU imIn, int x0, int y0, int x1, int y1) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(crop);

    int out_w = x1 - x0;
    int out_h = y1 - y0;
    if (out_w <= 0 || out_h <= 0) {
        return (ImagingGPU)ImagingError_ValueError("Invalid crop box");
    }

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, out_w, out_h);
    if (!out)
        return NULL;

    int err = _b->crop(_b, out, imIn, x0, y0, x1, y1);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU crop failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Expand(ImagingGPU imIn, int xmargin, int ymargin, const UINT8 *fill) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(expand);

    int out_w = imIn->xsize + 2 * xmargin;
    int out_h = imIn->ysize + 2 * ymargin;

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, out_w, out_h);
    if (!out)
        return NULL;

    int err = _b->expand(_b, out, imIn, xmargin, ymargin, fill);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU expand failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Offset(ImagingGPU imIn, int xoffset, int yoffset) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(offset);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->offset(_b, out, imIn, xoffset, yoffset);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU offset failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Negative(ImagingGPU imIn) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(negative);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->negative(_b, out, imIn);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU negative failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Posterize(ImagingGPU imIn, int bits) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(posterize);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->posterize(_b, out, imIn, bits);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU posterize failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Solarize(ImagingGPU imIn, int threshold) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(solarize);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->solarize(_b, out, imIn, threshold);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU solarize failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_Equalize(ImagingGPU imIn, const UINT8 *lut) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(equalize);

    ImagingGPU out = ImagingGPU_NewDirty(imIn->mode, imIn->xsize, imIn->ysize);
    if (!out)
        return NULL;

    int err = _b->equalize(_b, out, imIn, lut);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU equalize failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_LinearGradient(ModeID mode, int xsize, int ysize, int direction) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(linear_gradient);

    ImagingGPU out = ImagingGPU_NewDirty(mode, xsize, ysize);
    if (!out)
        return NULL;

    int err = _b->linear_gradient(_b, out, direction);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU linear_gradient failed");
    }
    return out;
}

ImagingGPU
ImagingGPU_RadialGradient(ModeID mode, int xsize, int ysize) {
    GET_BACKEND_OR_FAIL();
    CHECK_OP_OR_FAIL(radial_gradient);

    ImagingGPU out = ImagingGPU_NewDirty(mode, xsize, ysize);
    if (!out)
        return NULL;

    int err = _b->radial_gradient(_b, out);
    if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        return (ImagingGPU)ImagingError_ValueError("GPU radial_gradient failed");
    }
    return out;
}

int
ImagingGPU_Reduce(ImagingGPU out, ImagingGPU in, int factor_x, int factor_y) {
    GPUBackend _b = ImagingGPU_GetBackend();
    if (!_b)
        return GPU_ERROR_NO_BACKEND;
    if (!_b->reduce)
        return GPU_ERROR_UNSUPPORTED;
    return _b->reduce(_b, out, in, factor_x, factor_y);
}

int
ImagingGPU_RankFilter(ImagingGPU out, ImagingGPU in, int ksize, int rank) {
    GPUBackend _b = ImagingGPU_GetBackend();
    if (!_b)
        return GPU_ERROR_NO_BACKEND;
    if (!_b->rank_filter)
        return GPU_ERROR_UNSUPPORTED;
    return _b->rank_filter(_b, out, in, ksize, rank);
}

int
ImagingGPU_ModeFilter(ImagingGPU out, ImagingGPU in, int ksize) {
    GPUBackend _b = ImagingGPU_GetBackend();
    if (!_b)
        return GPU_ERROR_NO_BACKEND;
    if (!_b->mode_filter)
        return GPU_ERROR_UNSUPPORTED;
    return _b->mode_filter(_b, out, in, ksize);
}
