/*
 * Pillow GPU Acceleration Layer
 *
 * Abstract GPU backend supporting OpenCL and CUDA.
 * Images persist in VRAM between operations for maximum efficiency.
 *
 * Copyright (c) 2026 Pillow Contributors
 */

#ifndef __GPU_IMAGING_H__
#define __GPU_IMAGING_H__

#include "../Imaging.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* -------------------------------------------------------------------- */
/* Backend types                                                         */
/* -------------------------------------------------------------------- */

#define GPU_BACKEND_NONE 0
#define GPU_BACKEND_OPENCL 1
#define GPU_BACKEND_CUDA 2

/* Error codes */
#define GPU_OK 0
#define GPU_ERROR_MEMORY -1
#define GPU_ERROR_INVALID_ARG -2
#define GPU_ERROR_NO_BACKEND -3
#define GPU_ERROR_COMPILE -4
#define GPU_ERROR_LAUNCH -5
#define GPU_ERROR_UNSUPPORTED -6
#define GPU_ERROR_TRANSFER -7

/* Chop operation codes (matching PIL.ImageChops) */
#define GPU_CHOP_ADD 0
#define GPU_CHOP_SUBTRACT 1
#define GPU_CHOP_MULTIPLY 2
#define GPU_CHOP_SCREEN 3
#define GPU_CHOP_OVERLAY 4
#define GPU_CHOP_DIFFERENCE 5
#define GPU_CHOP_LIGHTER 6
#define GPU_CHOP_DARKER 7
#define GPU_CHOP_ADD_MODULO 8
#define GPU_CHOP_SUBTRACT_MODULO 9
#define GPU_CHOP_SOFT_LIGHT 10
#define GPU_CHOP_HARD_LIGHT 11
#define GPU_CHOP_AND 12
#define GPU_CHOP_OR 13
#define GPU_CHOP_XOR 14
#define GPU_CHOP_INVERT 15

/* Transpose operations (matching PIL.Image.Transpose) */
#define GPU_TRANSPOSE_FLIP_LR 0
#define GPU_TRANSPOSE_FLIP_TB 1
#define GPU_TRANSPOSE_ROTATE_90 2
#define GPU_TRANSPOSE_ROTATE_180 3
#define GPU_TRANSPOSE_ROTATE_270 4
#define GPU_TRANSPOSE_TRANSPOSE 5
#define GPU_TRANSPOSE_TRANSVERSE 6

/* Resample filters (matching PIL.Image.Resampling) */
#define GPU_RESAMPLE_NEAREST 0
#define GPU_RESAMPLE_LANCZOS 1
#define GPU_RESAMPLE_BILINEAR 2
#define GPU_RESAMPLE_BICUBIC 3
#define GPU_RESAMPLE_BOX 4
#define GPU_RESAMPLE_HAMMING 5

/* -------------------------------------------------------------------- */
/* GPU Buffer Handle                                                     */
/* -------------------------------------------------------------------- */

typedef struct GPUBufferInstance {
    size_t size; /* Buffer size in bytes */
    union {
        void *cl_mem;              /* OpenCL: cl_mem handle */
        unsigned long long cu_ptr; /* CUDA: CUdeviceptr */
    } handle;
} GPUBufferInstance;

typedef GPUBufferInstance *GPUBuffer;

/* -------------------------------------------------------------------- */
/* GPU Image (image stored in VRAM)                                     */
/* -------------------------------------------------------------------- */

typedef struct ImagingGPUInstance *ImagingGPU;

struct ImagingGPUInstance {
    /* Format (mirrors ImagingMemoryInstance) */
    ModeID mode;
    int type;  /* IMAGING_TYPE_UINT8, INT32, FLOAT32, SPECIAL */
    int bands; /* 1-4 */
    int xsize;
    int ysize;
    int pixelsize; /* 1, 2, or 4 */
    int linesize;  /* xsize * pixelsize */

    /* GPU storage */
    GPUBufferInstance buffer; /* VRAM buffer (contiguous, ysize * linesize) */
    int backend_type;         /* GPU_BACKEND_OPENCL or GPU_BACKEND_CUDA */

    /* Palette (for P mode) */
    ImagingPalette palette;
};

/* -------------------------------------------------------------------- */
/* GPU Backend (abstract interface via function pointers)               */
/* -------------------------------------------------------------------- */

typedef struct GPUBackendInstance *GPUBackend;

struct GPUBackendInstance {
    int type;              /* GPU_BACKEND_OPENCL or GPU_BACKEND_CUDA */
    const char *name;      /* "OpenCL" or "CUDA" */
    char device_name[256]; /* GPU device name */
    size_t max_mem_alloc;  /* Max single allocation */
    size_t total_mem;      /* Total device memory */

    /* Lifecycle */
    void (*shutdown)(GPUBackend self);

    /* Memory management */
    int (*buffer_alloc)(GPUBackend self, GPUBuffer buf, size_t size);
    void (*buffer_free)(GPUBackend self, GPUBuffer buf);
    int (*buffer_upload)(GPUBackend self, GPUBuffer buf, const void *data, size_t size);
    int (*buffer_download)(GPUBackend self, GPUBuffer buf, void *data, size_t size);
    int (*buffer_copy)(GPUBackend self, GPUBuffer dst, GPUBuffer src, size_t size);

    /* Core operations (all operate on VRAM buffers) */

    /* Blur / Filter */
    int (*gaussian_blur)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        float xradius,
        float yradius,
        int passes
    );
    int (*box_blur)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        float xradius,
        float yradius,
        int n
    );
    int (*unsharp_mask)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        float radius,
        int percent,
        int threshold
    );
    int (*filter)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        int ksize_x,
        int ksize_y,
        const float *kernel,
        float divisor,
        float offset
    );

    /* Resample */
    int (*resample)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        int xsize,
        int ysize,
        int filter,
        const float box[4]
    );

    /* Color conversion */
    int (*convert)(GPUBackend self, ImagingGPU out, ImagingGPU in, ModeID to_mode);

    /* Compositing */
    int (*blend)(
        GPUBackend self, ImagingGPU out, ImagingGPU im1, ImagingGPU im2, float alpha
    );
    int (*alpha_composite)(
        GPUBackend self, ImagingGPU out, ImagingGPU im1, ImagingGPU im2
    );

    /* Channel operations */
    int (*chop)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU im1,
        ImagingGPU im2,
        int op,
        float scale,
        int offset
    );

    /* Geometry */
    int (*transpose)(GPUBackend self, ImagingGPU out, ImagingGPU in, int op);
    int (*transform)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        int method,
        double a[8],
        int filter,
        int fill
    );

    /* Point / LUT */
    int (*point_lut)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, const UINT8 *lut, int bands
    );
    int (*point_transform)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, double scale, double offset
    );

    /* Color matrix */
    int (*color_matrix)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        const float *matrix,
        int ncolumns
    );

    /* Color LUT 3D */
    int (*color_lut_3d)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        int table_channels,
        int size1D,
        int size2D,
        int size3D,
        const INT16 *table
    );

    /* Fill */
    int (*fill)(GPUBackend self, ImagingGPU im, const void *color);

    /* Paste */
    int (*paste)(
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
    );

    /* Statistics */
    int (*histogram)(GPUBackend self, ImagingGPU im, long *hist_out);
    int (*getbbox)(GPUBackend self, ImagingGPU im, int bbox[4], int alpha_only);
    int (*getextrema)(GPUBackend self, ImagingGPU im, void *extrema);

    /* Copy */
    int (*copy)(GPUBackend self, ImagingGPU out, ImagingGPU in);

    /* Band operations */
    int (*getband)(GPUBackend self, ImagingGPU out, ImagingGPU in, int band);
    int (*putband)(GPUBackend self, ImagingGPU im, ImagingGPU band_im, int band);
    int (*fillband)(GPUBackend self, ImagingGPU im, int band, int color);
    int (*merge)(GPUBackend self, ImagingGPU out, ImagingGPU bands[4], int nbands);
    int (*split)(GPUBackend self, ImagingGPU im, ImagingGPU bands[4]);

    /* Effect */
    int (*effect_spread)(GPUBackend self, ImagingGPU out, ImagingGPU in, int distance);

    /* Extended operations */
    int (*crop)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, int x0, int y0, int x1, int y1
    );
    int (*expand)(
        GPUBackend self,
        ImagingGPU out,
        ImagingGPU in,
        int xmargin,
        int ymargin,
        const UINT8 *fill
    );
    int (*offset)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, int xoffset, int yoffset
    );
    int (*negative)(GPUBackend self, ImagingGPU out, ImagingGPU in);
    int (*posterize)(GPUBackend self, ImagingGPU out, ImagingGPU in, int bits);
    int (*solarize)(GPUBackend self, ImagingGPU out, ImagingGPU in, int threshold);
    int (*equalize)(GPUBackend self, ImagingGPU out, ImagingGPU in, const UINT8 *lut);
    int (*linear_gradient)(GPUBackend self, ImagingGPU out, int direction);
    int (*radial_gradient)(GPUBackend self, ImagingGPU out);

    /* New parallel kernels */
    int (*reduce)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, int factor_x, int factor_y
    );
    int (*rank_filter)(
        GPUBackend self, ImagingGPU out, ImagingGPU in, int ksize, int rank
    );
    int (*mode_filter)(GPUBackend self, ImagingGPU out, ImagingGPU in, int ksize);

    /* Backend-specific context (OpenCL state or CUDA state) */
    void *ctx;
};

/* -------------------------------------------------------------------- */
/* Public API                                                           */
/* -------------------------------------------------------------------- */

/* Backend initialization */
extern GPUBackend
ImagingGPU_BackendInit(int preferred_type);

extern GPUBackend
ImagingGPU_GetBackend(void);

extern void
ImagingGPU_BackendShutdown(void);

extern int
ImagingGPU_IsAvailable(void);

extern const char *
ImagingGPU_GetBackendName(void);

extern const char *
ImagingGPU_GetDeviceName(void);

/* GPU Image lifecycle */
extern ImagingGPU
ImagingGPU_New(ModeID mode, int xsize, int ysize);

extern ImagingGPU
ImagingGPU_NewDirty(ModeID mode, int xsize, int ysize);

extern void
ImagingGPU_Delete(ImagingGPU im);

/* CPU ↔ GPU transfer */
extern ImagingGPU
ImagingGPU_FromImaging(Imaging im);

extern ImagingGPU
ImagingGPU_FromBytes(
    ModeID mode, int xsize, int ysize, const void *data, size_t data_len
);

extern Imaging
ImagingGPU_ToImaging(ImagingGPU gpu_im);

extern int
ImagingGPU_DownloadRaw(ImagingGPU gpu_im, void *out_buf, size_t buf_size);

extern int
ImagingGPU_Upload(ImagingGPU gpu_im, Imaging im);

extern int
ImagingGPU_Download(ImagingGPU gpu_im, Imaging im);

/* GPU operations (dispatch to active backend) */

extern ImagingGPU
ImagingGPU_GaussianBlur(ImagingGPU imIn, float xradius, float yradius, int passes);

extern ImagingGPU
ImagingGPU_BoxBlur(ImagingGPU imIn, float xradius, float yradius, int n);

extern ImagingGPU
ImagingGPU_UnsharpMask(ImagingGPU imIn, float radius, int percent, int threshold);

extern ImagingGPU
ImagingGPU_Filter(
    ImagingGPU imIn,
    int ksize_x,
    int ksize_y,
    const float *kernel,
    float divisor,
    float offset
);

extern ImagingGPU
ImagingGPU_Resample(ImagingGPU imIn, int xsize, int ysize, int filter, float box[4]);

extern ImagingGPU
ImagingGPU_Convert(ImagingGPU imIn, ModeID to_mode);

extern ImagingGPU
ImagingGPU_Blend(ImagingGPU im1, ImagingGPU im2, float alpha);

extern ImagingGPU
ImagingGPU_AlphaComposite(ImagingGPU im1, ImagingGPU im2);

extern ImagingGPU
ImagingGPU_Chop(ImagingGPU im1, ImagingGPU im2, int op, float scale, int offset);

extern ImagingGPU
ImagingGPU_Transpose(ImagingGPU imIn, int op);

extern ImagingGPU
ImagingGPU_Transform(
    ImagingGPU imIn, int xsize, int ysize, int method, double a[8], int filter, int fill
);

extern ImagingGPU
ImagingGPU_PointLut(ImagingGPU imIn, const UINT8 *lut, int bands);

extern ImagingGPU
ImagingGPU_PointTransform(ImagingGPU imIn, double scale, double offset);

extern ImagingGPU
ImagingGPU_ColorMatrix(
    ImagingGPU imIn, ModeID out_mode, const float *matrix, int ncolumns
);

extern ImagingGPU
ImagingGPU_ColorLUT3D(
    ImagingGPU imIn,
    int table_channels,
    int size1D,
    int size2D,
    int size3D,
    const INT16 *table
);

extern ImagingGPU
ImagingGPU_Fill(ModeID mode, int xsize, int ysize, const void *color);

extern int
ImagingGPU_PasteInPlace(
    ImagingGPU dest, ImagingGPU src, ImagingGPU mask, int dx, int dy
);

extern ImagingGPU
ImagingGPU_Copy(ImagingGPU imIn);

extern ImagingGPU
ImagingGPU_GetBand(ImagingGPU imIn, int band);

extern ImagingGPU
ImagingGPU_PutBand(ImagingGPU imIn, ImagingGPU band_im, int band);

extern ImagingGPU
ImagingGPU_FillBand(ImagingGPU imIn, int band, int color);

extern ImagingGPU
ImagingGPU_Merge(ModeID mode, ImagingGPU bands[4]);

extern int
ImagingGPU_Split(ImagingGPU imIn, ImagingGPU bands[4]);

extern int
ImagingGPU_Histogram(ImagingGPU imIn, long *hist_out);

extern int
ImagingGPU_GetBBox(ImagingGPU imIn, int bbox[4], int alpha_only);

extern int
ImagingGPU_GetExtrema(ImagingGPU imIn, void *extrema);

extern ImagingGPU
ImagingGPU_EffectSpread(ImagingGPU imIn, int distance);

/* Extended operations */
extern ImagingGPU
ImagingGPU_Crop(ImagingGPU imIn, int x0, int y0, int x1, int y1);

extern ImagingGPU
ImagingGPU_Expand(ImagingGPU imIn, int xmargin, int ymargin, const UINT8 *fill);

extern ImagingGPU
ImagingGPU_Offset(ImagingGPU imIn, int xoffset, int yoffset);

extern ImagingGPU
ImagingGPU_Negative(ImagingGPU imIn);

extern ImagingGPU
ImagingGPU_Posterize(ImagingGPU imIn, int bits);

extern ImagingGPU
ImagingGPU_Solarize(ImagingGPU imIn, int threshold);

extern ImagingGPU
ImagingGPU_Equalize(ImagingGPU imIn, const UINT8 *lut);

extern ImagingGPU
ImagingGPU_LinearGradient(ModeID mode, int xsize, int ysize, int direction);

extern ImagingGPU
ImagingGPU_RadialGradient(ModeID mode, int xsize, int ysize);

extern int
ImagingGPU_Reduce(ImagingGPU out, ImagingGPU in, int factor_x, int factor_y);

extern int
ImagingGPU_RankFilter(ImagingGPU out, ImagingGPU in, int ksize, int rank);

extern int
ImagingGPU_ModeFilter(ImagingGPU out, ImagingGPU in, int ksize);

/* Backend-specific initialization (called by ImagingGPU_BackendInit) */
#ifdef HAVE_OPENCL
extern GPUBackend
ImagingGPU_OpenCL_Init(void);
#endif

#ifdef HAVE_CUDA
extern GPUBackend
ImagingGPU_CUDA_Init(void);
#endif

/* Utility */
extern int
ImagingGPU_ModeToPixelSize(ModeID mode);

extern int
ImagingGPU_ModeToType(ModeID mode);

extern int
ImagingGPU_ModeToBands(ModeID mode);

#if defined(__cplusplus)
}
#endif

#endif /* __GPU_IMAGING_H__ */
