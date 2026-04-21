/*
 * Pillow GPU Acceleration — CPython Extension Module
 *
 * Exposes GPU imaging operations to Python as PIL._imaging_gpu.
 * Wraps ImagingGPU objects and dispatches to the active GPU backend.
 *
 * Copyright (c) 2026 Pillow Contributors
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "libImaging/gpu/GpuImaging.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stddef.h>

/* -------------------------------------------------------------------- */
/* ImagingError_* functions (these are CPython-specific, same as         */
/* _imaging.c; needed by libImaging functions linked into this module)   */
/* -------------------------------------------------------------------- */

void *
ImagingError_MemoryError(void) {
    return PyErr_NoMemory();
}

void *
ImagingError_Mismatch(void) {
    PyErr_SetString(PyExc_ValueError, "images do not match");
    return NULL;
}

void *
ImagingError_ModeError(void) {
    PyErr_SetString(PyExc_ValueError, "image has wrong mode");
    return NULL;
}

void *
ImagingError_ValueError(const char *message) {
    PyErr_SetString(
        PyExc_ValueError, (message) ? (char *)message : "unrecognized argument value"
    );
    return NULL;
}

/* -------------------------------------------------------------------- */
/* OBJECT ADMINISTRATION                                                  */
/* -------------------------------------------------------------------- */

typedef struct {
    PyObject_HEAD ImagingGPU gpu_image;
} ImagingGPUObject;

static PyTypeObject ImagingGPU_Type;

/* Forward declarations for CPU Image type from _imaging */
/* We use capsule / pointer exchange to avoid depending on _imaging.c internals */

#define PyImagingGPU_Check(op) (Py_TYPE(op) == &ImagingGPU_Type)

static void
_gpu_dealloc(ImagingGPUObject *self) {
    if (self->gpu_image) {
        ImagingGPU_Delete(self->gpu_image);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static ImagingGPUObject *
_gpu_wrap(ImagingGPU gpu_im) {
    if (!gpu_im) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "GPU operation failed");
        }
        return NULL;
    }
    ImagingGPUObject *obj = PyObject_New(ImagingGPUObject, &ImagingGPU_Type);
    if (!obj) {
        ImagingGPU_Delete(gpu_im);
        return NULL;
    }
    obj->gpu_image = gpu_im;
    return obj;
}

/* Get ImagingGPU from a Python object (checks type) */
static ImagingGPU
_gpu_get(PyObject *obj) {
    if (!PyImagingGPU_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected a GPU image object");
        return NULL;
    }
    return ((ImagingGPUObject *)obj)->gpu_image;
}

/* Helper: resolve ModeID from Python string */
static ModeID
_gpu_parse_mode(PyObject *mode_obj) {
    const char *mode_str = PyUnicode_AsUTF8(mode_obj);
    if (!mode_str)
        return IMAGING_MODE_UNKNOWN;

    /* Match mode strings to ModeID enum values */
    if (strcmp(mode_str, "1") == 0)
        return IMAGING_MODE_1;
    if (strcmp(mode_str, "L") == 0)
        return IMAGING_MODE_L;
    if (strcmp(mode_str, "LA") == 0)
        return IMAGING_MODE_LA;
    if (strcmp(mode_str, "I") == 0)
        return IMAGING_MODE_I;
    if (strcmp(mode_str, "F") == 0)
        return IMAGING_MODE_F;
    if (strcmp(mode_str, "P") == 0)
        return IMAGING_MODE_P;
    if (strcmp(mode_str, "PA") == 0)
        return IMAGING_MODE_PA;
    if (strcmp(mode_str, "RGB") == 0)
        return IMAGING_MODE_RGB;
    if (strcmp(mode_str, "RGBA") == 0)
        return IMAGING_MODE_RGBA;
    if (strcmp(mode_str, "RGBX") == 0)
        return IMAGING_MODE_RGBX;
    if (strcmp(mode_str, "CMYK") == 0)
        return IMAGING_MODE_CMYK;
    if (strcmp(mode_str, "YCbCr") == 0)
        return IMAGING_MODE_YCbCr;
    if (strcmp(mode_str, "LAB") == 0)
        return IMAGING_MODE_LAB;
    if (strcmp(mode_str, "HSV") == 0)
        return IMAGING_MODE_HSV;
    if (strcmp(mode_str, "I;16") == 0)
        return IMAGING_MODE_I_16;
    if (strcmp(mode_str, "I;16L") == 0)
        return IMAGING_MODE_I_16L;
    if (strcmp(mode_str, "I;16B") == 0)
        return IMAGING_MODE_I_16B;
    if (strcmp(mode_str, "I;16N") == 0)
        return IMAGING_MODE_I_16N;

    PyErr_Format(PyExc_ValueError, "unsupported mode '%s'", mode_str);
    return IMAGING_MODE_UNKNOWN;
}

/* -------------------------------------------------------------------- */
/* MODULE-LEVEL FUNCTIONS                                                */
/* -------------------------------------------------------------------- */

static PyObject *
_gpu_backend_init(PyObject *self, PyObject *args) {
    int preferred_type = 0; /* 0 = auto */
    if (!PyArg_ParseTuple(args, "|i", &preferred_type)) {
        return NULL;
    }

    GPUBackend backend;
    Py_BEGIN_ALLOW_THREADS backend = ImagingGPU_BackendInit(preferred_type);
    Py_END_ALLOW_THREADS

        if (!backend) {
        PyErr_SetString(
            PyExc_RuntimeError, "No GPU backend available (OpenCL/CUDA not found)"
        );
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gpu_backend_shutdown(PyObject *self, PyObject *args) {
    (void)args;
    ImagingGPU_BackendShutdown();
    Py_RETURN_NONE;
}

static PyObject *
_gpu_is_available(PyObject *self, PyObject *args) {
    (void)args;
    if (ImagingGPU_IsAvailable()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
_gpu_get_backend_name(PyObject *self, PyObject *args) {
    (void)args;
    const char *name = ImagingGPU_GetBackendName();
    if (!name) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(name);
}

static PyObject *
_gpu_get_device_name(PyObject *self, PyObject *args) {
    (void)args;
    const char *name = ImagingGPU_GetDeviceName();
    if (!name) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(name);
}

static PyObject *
_gpu_new(PyObject *self, PyObject *args) {
    PyObject *mode_obj;
    int xsize, ysize;
    if (!PyArg_ParseTuple(args, "O(ii)", &mode_obj, &xsize, &ysize)) {
        return NULL;
    }
    ModeID mode = _gpu_parse_mode(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN)
        return NULL;

    ImagingGPU im;
    Py_BEGIN_ALLOW_THREADS im = ImagingGPU_New(mode, xsize, ysize);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(im);
}

static PyObject *
_gpu_fill(PyObject *self, PyObject *args) {
    PyObject *mode_obj;
    int xsize, ysize;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (!PyArg_ParseTuple(
            args, "O(ii)|(iiii)", &mode_obj, &xsize, &ysize, &c0, &c1, &c2, &c3
        )) {
        return NULL;
    }
    ModeID mode = _gpu_parse_mode(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN)
        return NULL;

    UINT8 color[4] = {(UINT8)c0, (UINT8)c1, (UINT8)c2, (UINT8)c3};

    ImagingGPU im;
    Py_BEGIN_ALLOW_THREADS im = ImagingGPU_Fill(mode, xsize, ysize, color);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(im);
}

static PyObject *
_gpu_from_imaging(PyObject *self, PyObject *args) {
    /* Takes a PyCapsule wrapping an Imaging (from _imaging) */
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    Imaging im = NULL;

    if (PyCapsule_IsValid(capsule, "PIL Imaging")) {
        im = (Imaging)PyCapsule_GetPointer(capsule, "PIL Imaging");
    } else if (PyCapsule_IsValid(capsule, "Imaging")) {
        im = (Imaging)PyCapsule_GetPointer(capsule, "Imaging");
    } else {
        /* Try via ptr attribute (ImagingObject.im.ptr returns a PyCapsule) */
        PyObject *ptr_attr = PyObject_GetAttrString(capsule, "ptr");
        if (ptr_attr) {
            if (PyCapsule_IsValid(ptr_attr, "PIL Imaging")) {
                im = (Imaging)PyCapsule_GetPointer(ptr_attr, "PIL Imaging");
            } else {
                im = (Imaging)PyLong_AsVoidPtr(ptr_attr);
            }
            Py_DECREF(ptr_attr);
        } else {
            PyErr_Clear();
            /* Try id attribute from ImagingObject */
            PyObject *id_attr = PyObject_GetAttrString(capsule, "id");
            if (id_attr) {
                im = (Imaging)PyLong_AsVoidPtr(id_attr);
                Py_DECREF(id_attr);
            } else {
                PyErr_SetString(
                    PyExc_TypeError,
                    "expected an Imaging capsule, object with .ptr, or .id attribute"
                );
                return NULL;
            }
        }
    }

    if (!im) {
        PyErr_SetString(PyExc_ValueError, "NULL Imaging pointer");
        return NULL;
    }

    ImagingGPU gpu_im;
    Py_BEGIN_ALLOW_THREADS gpu_im = ImagingGPU_FromImaging(im);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(gpu_im);
}

static PyObject *
_gpu_from_bytes(PyObject *self, PyObject *args) {
    /* from_bytes(mode_str, (w, h), data_bytes) */
    (void)self;
    const char *mode_str;
    int xsize, ysize;
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "s(ii)y*", &mode_str, &xsize, &ysize, &buf)) {
        return NULL;
    }

    ModeID mode = findModeID(mode_str);
    if (mode == IMAGING_MODE_UNKNOWN) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "unknown mode");
        return NULL;
    }

    ImagingGPU gpu_im;
    Py_BEGIN_ALLOW_THREADS gpu_im =
        ImagingGPU_FromBytes(mode, xsize, ysize, buf.buf, (size_t)buf.len);
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&buf);
    return (PyObject *)_gpu_wrap(gpu_im);
}

static PyObject *
_gpu_to_bytes(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    ImagingGPU gpu_im = self->gpu_image;
    if (!gpu_im) {
        PyErr_SetString(PyExc_RuntimeError, "NULL GPU image");
        return NULL;
    }
    size_t total_size = (size_t)gpu_im->linesize * (size_t)gpu_im->ysize;
    PyObject *bytes_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
    if (!bytes_obj) {
        return NULL;
    }
    char *out = PyBytes_AS_STRING(bytes_obj);
    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_DownloadRaw(gpu_im, out, total_size);
    Py_END_ALLOW_THREADS if (err != GPU_OK) {
        Py_DECREF(bytes_obj);
        PyErr_SetString(PyExc_RuntimeError, "GPU download failed");
        return NULL;
    }
    return bytes_obj;
}

static PyObject *
_gpu_blend(PyObject *self, PyObject *args) {
    PyObject *o1, *o2;
    float alpha;
    if (!PyArg_ParseTuple(args, "OOf", &o1, &o2, &alpha)) {
        return NULL;
    }
    ImagingGPU im1 = _gpu_get(o1);
    ImagingGPU im2 = _gpu_get(o2);
    if (!im1 || !im2)
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Blend(im1, im2, alpha);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_alpha_composite(PyObject *self, PyObject *args) {
    PyObject *o1, *o2;
    if (!PyArg_ParseTuple(args, "OO", &o1, &o2)) {
        return NULL;
    }
    ImagingGPU im1 = _gpu_get(o1);
    ImagingGPU im2 = _gpu_get(o2);
    if (!im1 || !im2)
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_AlphaComposite(im1, im2);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

/* -------------------------------------------------------------------- */
/* OBJECT METHODS                                                        */
/* -------------------------------------------------------------------- */

static PyObject *
_gpu_to_imaging(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    Imaging im;
    Py_BEGIN_ALLOW_THREADS im = ImagingGPU_ToImaging(self->gpu_image);
    Py_END_ALLOW_THREADS

        if (!im) {
        PyErr_SetString(PyExc_RuntimeError, "GPU->CPU transfer failed");
        return NULL;
    }

    /* Return as PyCapsule for interop with PIL.Image */
    return PyCapsule_New(im, "PIL Imaging", NULL);
}

static PyObject *
_gpu_gaussian_blur(ImagingGPUObject *self, PyObject *args) {
    float xradius, yradius;
    int passes = 3;
    if (!PyArg_ParseTuple(args, "ff|i", &xradius, &yradius, &passes)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_GaussianBlur(self->gpu_image, xradius, yradius, passes);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_box_blur(ImagingGPUObject *self, PyObject *args) {
    float xradius, yradius;
    int n = 1;
    if (!PyArg_ParseTuple(args, "ff|i", &xradius, &yradius, &n)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_BoxBlur(self->gpu_image, xradius, yradius, n);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_unsharp_mask(ImagingGPUObject *self, PyObject *args) {
    float radius;
    int percent, threshold;
    if (!PyArg_ParseTuple(args, "fii", &radius, &percent, &threshold)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_UnsharpMask(self->gpu_image, radius, percent, threshold);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_filter(ImagingGPUObject *self, PyObject *args) {
    int ksize_x, ksize_y;
    float divisor, offset;
    PyObject *kernel_seq;
    if (!PyArg_ParseTuple(
            args, "iiOff", &ksize_x, &ksize_y, &kernel_seq, &divisor, &offset
        )) {
        return NULL;
    }

    Py_ssize_t klen = PySequence_Length(kernel_seq);
    if (klen != ksize_x * ksize_y) {
        PyErr_SetString(PyExc_ValueError, "kernel size mismatch");
        return NULL;
    }

    float *kernel = (float *)malloc(klen * sizeof(float));
    if (!kernel)
        return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < klen; i++) {
        PyObject *item = PySequence_GetItem(kernel_seq, i);
        if (!item) {
            free(kernel);
            return NULL;
        }
        kernel[i] = (float)PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            free(kernel);
            return NULL;
        }
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Filter(self->gpu_image, ksize_x, ksize_y, kernel, divisor, offset);
    Py_END_ALLOW_THREADS

        free(kernel);
    return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_resize(ImagingGPUObject *self, PyObject *args) {
    int xsize, ysize, filter = GPU_RESAMPLE_BILINEAR;
    float box[4] = {0, 0, 0, 0};
    if (!PyArg_ParseTuple(
            args,
            "(ii)|i(ffff)",
            &xsize,
            &ysize,
            &filter,
            &box[0],
            &box[1],
            &box[2],
            &box[3]
        )) {
        return NULL;
    }
    /* Default box = full source */
    if (box[2] == 0 && box[3] == 0) {
        box[2] = (float)self->gpu_image->xsize;
        box[3] = (float)self->gpu_image->ysize;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Resample(self->gpu_image, xsize, ysize, filter, box);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_convert(ImagingGPUObject *self, PyObject *args) {
    PyObject *mode_obj;
    if (!PyArg_ParseTuple(args, "O", &mode_obj)) {
        return NULL;
    }
    ModeID mode = _gpu_parse_mode(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN)
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Convert(self->gpu_image, mode);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_transpose(ImagingGPUObject *self, PyObject *args) {
    int op;
    if (!PyArg_ParseTuple(args, "i", &op)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Transpose(self->gpu_image, op);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_transform(ImagingGPUObject *self, PyObject *args) {
    int xsize, ysize, method, filter = 0, fill = 1;
    double a[8] = {0};
    if (!PyArg_ParseTuple(
            args,
            "(ii)i(dddddddd)|ii",
            &xsize,
            &ysize,
            &method,
            &a[0],
            &a[1],
            &a[2],
            &a[3],
            &a[4],
            &a[5],
            &a[6],
            &a[7],
            &filter,
            &fill
        )) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Transform(self->gpu_image, xsize, ysize, method, a, filter, fill);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_chop(ImagingGPUObject *self, PyObject *args) {
    PyObject *other_obj;
    int op;
    float scale = 1.0f;
    int offset = 0;
    if (!PyArg_ParseTuple(args, "Oi|fi", &other_obj, &op, &scale, &offset)) {
        return NULL;
    }
    ImagingGPU other = _gpu_get(other_obj);
    if (!other)
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Chop(self->gpu_image, other, op, scale, offset);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_point_transform(ImagingGPUObject *self, PyObject *args) {
    double scale, offset;
    if (!PyArg_ParseTuple(args, "dd", &scale, &offset)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_PointTransform(self->gpu_image, scale, offset);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_copy(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Copy(self->gpu_image);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_getband(ImagingGPUObject *self, PyObject *args) {
    int band;
    if (!PyArg_ParseTuple(args, "i", &band)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_GetBand(self->gpu_image, band);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_putband(ImagingGPUObject *self, PyObject *args) {
    PyObject *band_obj;
    int band;
    if (!PyArg_ParseTuple(args, "Oi", &band_obj, &band)) {
        return NULL;
    }
    ImagingGPU band_im = _gpu_get(band_obj);
    if (!band_im)
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_PutBand(self->gpu_image, band_im, band);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_fillband(ImagingGPUObject *self, PyObject *args) {
    int band, color;
    if (!PyArg_ParseTuple(args, "ii", &band, &color)) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_FillBand(self->gpu_image, band, color);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_histogram(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    int bands = self->gpu_image->bands;
    int hist_size = bands * 256;
    long *hist = (long *)calloc(hist_size, sizeof(long));
    if (!hist)
        return PyErr_NoMemory();

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_Histogram(self->gpu_image, hist);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        free(hist);
        PyErr_SetString(PyExc_RuntimeError, "GPU histogram failed");
        return NULL;
    }

    PyObject *list = PyList_New(hist_size);
    if (!list) {
        free(hist);
        return NULL;
    }
    for (int i = 0; i < hist_size; i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(hist[i]));
    }
    free(hist);
    return list;
}

/* -------------------------------------------------------------------- */
/* NEW METHODS: point_lut, paste, merge, split, getbbox, getextrema,    */
/* effect_spread, crop, expand, offset, negative, posterize, solarize,  */
/* equalize                                                              */
/* -------------------------------------------------------------------- */

static PyObject *
_gpu_point_lut(ImagingGPUObject *self, PyObject *args) {
    Py_buffer lut_buf;
    int bands;
    if (!PyArg_ParseTuple(args, "y*i", &lut_buf, &bands))
        return NULL;

    if (lut_buf.len < bands * 256) {
        PyBuffer_Release(&lut_buf);
        PyErr_SetString(PyExc_ValueError, "LUT too small");
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_PointLut(self->gpu_image, (const UINT8 *)lut_buf.buf, bands);
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&lut_buf);
    return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_paste(ImagingGPUObject *self, PyObject *args) {
    PyObject *src_obj, *mask_obj = Py_None;
    int dx = 0, dy = 0;
    if (!PyArg_ParseTuple(args, "O|(ii)O", &src_obj, &dx, &dy, &mask_obj))
        return NULL;

    ImagingGPU src_im = _gpu_get(src_obj);
    if (!src_im)
        return NULL;

    ImagingGPU mask_im = NULL;
    if (mask_obj != Py_None) {
        if (!PyImagingGPU_Check(mask_obj)) {
            PyErr_SetString(PyExc_TypeError, "mask must be a GPU image");
            return NULL;
        }
        mask_im = ((ImagingGPUObject *)mask_obj)->gpu_image;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS err =
        ImagingGPU_PasteInPlace(self->gpu_image, src_im, mask_im, dx, dy);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        PyErr_SetString(PyExc_RuntimeError, "GPU paste failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gpu_merge(PyObject *self_unused, PyObject *args) {
    (void)self_unused;
    const char *mode_str;
    PyObject *band_list;
    if (!PyArg_ParseTuple(args, "sO", &mode_str, &band_list))
        return NULL;

    if (!PyList_Check(band_list) && !PyTuple_Check(band_list)) {
        PyErr_SetString(PyExc_TypeError, "bands must be a list or tuple");
        return NULL;
    }

    Py_ssize_t nbands = PySequence_Size(band_list);
    if (nbands < 1 || nbands > 4) {
        PyErr_SetString(PyExc_ValueError, "need 1-4 bands");
        return NULL;
    }

    PyObject *mode_obj = PyUnicode_FromString(mode_str);
    ModeID mode = _gpu_parse_mode(mode_obj);
    Py_DECREF(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN) {
        return NULL;
    }

    ImagingGPU bands[4] = {NULL, NULL, NULL, NULL};
    for (Py_ssize_t i = 0; i < nbands; i++) {
        PyObject *item = PySequence_GetItem(band_list, i);
        bands[i] = _gpu_get(item);
        Py_DECREF(item);
        if (!bands[i])
            return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Merge(mode, bands);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_split_method(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    int nbands = self->gpu_image->bands;
    ImagingGPU bands[4] = {NULL, NULL, NULL, NULL};

    /* Allocate output band images */
    for (int i = 0; i < nbands; i++) {
        bands[i] = ImagingGPU_NewDirty(
            IMAGING_MODE_L, self->gpu_image->xsize, self->gpu_image->ysize
        );
        if (!bands[i]) {
            for (int j = 0; j < i; j++) ImagingGPU_Delete(bands[j]);
            return PyErr_NoMemory();
        }
    }

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_Split(self->gpu_image, bands);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        for (int i = 0; i < nbands; i++) ImagingGPU_Delete(bands[i]);
        PyErr_SetString(PyExc_RuntimeError, "GPU split failed");
        return NULL;
    }

    PyObject *tuple = PyTuple_New(nbands);
    for (int i = 0; i < nbands; i++) {
        PyTuple_SET_ITEM(tuple, i, (PyObject *)_gpu_wrap(bands[i]));
    }
    return tuple;
}

static PyObject *
_gpu_getbbox(ImagingGPUObject *self, PyObject *args) {
    int alpha_only = 0;
    if (!PyArg_ParseTuple(args, "|i", &alpha_only))
        return NULL;

    int bbox[4] = {0, 0, 0, 0};

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_GetBBox(self->gpu_image, bbox, alpha_only);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        PyErr_SetString(PyExc_RuntimeError, "GPU getbbox failed");
        return NULL;
    }

    if (bbox[0] == 0 && bbox[1] == 0 && bbox[2] == 0 && bbox[3] == 0) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("(iiii)", bbox[0], bbox[1], bbox[2], bbox[3]);
}

static PyObject *
_gpu_getextrema(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    int bands = self->gpu_image->bands;
    UINT8 extrema[8]; /* max 4 bands * 2 (min, max) */
    memset(extrema, 0, sizeof(extrema));

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_GetExtrema(self->gpu_image, extrema);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        PyErr_SetString(PyExc_RuntimeError, "GPU getextrema failed");
        return NULL;
    }

    if (bands == 1) {
        return Py_BuildValue("(ii)", (int)extrema[0], (int)extrema[1]);
    }
    PyObject *tuple = PyTuple_New(bands);
    for (int i = 0; i < bands; i++) {
        PyTuple_SET_ITEM(
            tuple,
            i,
            Py_BuildValue("(ii)", (int)extrema[i * 2], (int)extrema[i * 2 + 1])
        );
    }
    return tuple;
}

static PyObject *
_gpu_effect_spread(ImagingGPUObject *self, PyObject *args) {
    int distance;
    if (!PyArg_ParseTuple(args, "i", &distance))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_EffectSpread(self->gpu_image, distance);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_crop(ImagingGPUObject *self, PyObject *args) {
    int x0, y0, x1, y1;
    if (!PyArg_ParseTuple(args, "(iiii)", &x0, &y0, &x1, &y1))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Crop(self->gpu_image, x0, y0, x1, y1);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_expand(ImagingGPUObject *self, PyObject *args) {
    int xmargin, ymargin;
    int fill0 = 0, fill1 = 0, fill2 = 0, fill3 = 0;
    if (!PyArg_ParseTuple(
            args, "ii|iiii", &xmargin, &ymargin, &fill0, &fill1, &fill2, &fill3
        ))
        return NULL;

    UINT8 fill[4] = {(UINT8)fill0, (UINT8)fill1, (UINT8)fill2, (UINT8)fill3};

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Expand(self->gpu_image, xmargin, ymargin, fill);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_offset_method(ImagingGPUObject *self, PyObject *args) {
    int xoffset, yoffset = 0;
    if (!PyArg_ParseTuple(args, "i|i", &xoffset, &yoffset))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Offset(self->gpu_image, xoffset, yoffset);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_negative_method(ImagingGPUObject *self, PyObject *args) {
    (void)args;
    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Negative(self->gpu_image);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_posterize_method(ImagingGPUObject *self, PyObject *args) {
    int bits;
    if (!PyArg_ParseTuple(args, "i", &bits))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Posterize(self->gpu_image, bits);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_solarize_method(ImagingGPUObject *self, PyObject *args) {
    int threshold = 128;
    if (!PyArg_ParseTuple(args, "|i", &threshold))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_Solarize(self->gpu_image, threshold);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_equalize_method(ImagingGPUObject *self, PyObject *args) {
    Py_buffer lut_buf;
    if (!PyArg_ParseTuple(args, "y*", &lut_buf))
        return NULL;

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out =
        ImagingGPU_Equalize(self->gpu_image, (const UINT8 *)lut_buf.buf);
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&lut_buf);
    return (PyObject *)_gpu_wrap(out);
}

/* Module-level gradient functions */
static PyObject *
_gpu_linear_gradient(PyObject *self_unused, PyObject *args) {
    (void)self_unused;
    const char *mode_str;
    int w, h, direction = 0;
    if (!PyArg_ParseTuple(args, "s(ii)|i", &mode_str, &w, &h, &direction))
        return NULL;

    PyObject *mode_obj = PyUnicode_FromString(mode_str);
    ModeID mode = _gpu_parse_mode(mode_obj);
    Py_DECREF(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_LinearGradient(mode, w, h, direction);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_radial_gradient(PyObject *self_unused, PyObject *args) {
    (void)self_unused;
    const char *mode_str;
    int w, h;
    if (!PyArg_ParseTuple(args, "s(ii)", &mode_str, &w, &h))
        return NULL;

    PyObject *mode_obj = PyUnicode_FromString(mode_str);
    ModeID mode = _gpu_parse_mode(mode_obj);
    Py_DECREF(mode_obj);
    if (mode == IMAGING_MODE_UNKNOWN) {
        return NULL;
    }

    ImagingGPU out;
    Py_BEGIN_ALLOW_THREADS out = ImagingGPU_RadialGradient(mode, w, h);
    Py_END_ALLOW_THREADS

        return (PyObject *)_gpu_wrap(out);
}

/* -------------------------------------------------------------------- */
/* reduce / rank_filter / mode_filter                                    */
/* -------------------------------------------------------------------- */

static PyObject *
_gpu_reduce_method(ImagingGPUObject *self, PyObject *args) {
    int factor_x, factor_y;
    if (!PyArg_ParseTuple(args, "ii", &factor_x, &factor_y))
        return NULL;

    ImagingGPU im = self->gpu_image;
    int out_w = (im->xsize + factor_x - 1) / factor_x;
    int out_h = (im->ysize + factor_y - 1) / factor_y;

    ImagingGPU out = ImagingGPU_NewDirty(im->mode, out_w, out_h);
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "GPU reduce: alloc failed");
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_Reduce(out, im, factor_x, factor_y);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        PyErr_SetString(PyExc_RuntimeError, "GPU reduce failed");
        return NULL;
    }
    return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_rank_filter_method(ImagingGPUObject *self, PyObject *args) {
    int ksize, rank;
    if (!PyArg_ParseTuple(args, "ii", &ksize, &rank))
        return NULL;

    ImagingGPU im = self->gpu_image;
    ImagingGPU out = ImagingGPU_NewDirty(im->mode, im->xsize, im->ysize);
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "GPU rank_filter: alloc failed");
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_RankFilter(out, im, ksize, rank);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        PyErr_SetString(PyExc_RuntimeError, "GPU rank_filter failed");
        return NULL;
    }
    return (PyObject *)_gpu_wrap(out);
}

static PyObject *
_gpu_mode_filter_method(ImagingGPUObject *self, PyObject *args) {
    int ksize;
    if (!PyArg_ParseTuple(args, "i", &ksize))
        return NULL;

    ImagingGPU im = self->gpu_image;
    ImagingGPU out = ImagingGPU_NewDirty(im->mode, im->xsize, im->ysize);
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "GPU mode_filter: alloc failed");
        return NULL;
    }

    int err;
    Py_BEGIN_ALLOW_THREADS err = ImagingGPU_ModeFilter(out, im, ksize);
    Py_END_ALLOW_THREADS

        if (err != GPU_OK) {
        ImagingGPU_Delete(out);
        PyErr_SetString(PyExc_RuntimeError, "GPU mode_filter failed");
        return NULL;
    }
    return (PyObject *)_gpu_wrap(out);
}

/* -------------------------------------------------------------------- */
/* ATTRIBUTES                                                            */
/* -------------------------------------------------------------------- */

static PyObject *
_gpu_getattr_mode(ImagingGPUObject *self, void *closure) {
    (void)closure;
    const ModeData *md = getModeData(self->gpu_image->mode);
    return PyUnicode_FromString(md->name);
}

static PyObject *
_gpu_getattr_size(ImagingGPUObject *self, void *closure) {
    (void)closure;
    return Py_BuildValue("(ii)", self->gpu_image->xsize, self->gpu_image->ysize);
}

static PyObject *
_gpu_getattr_bands(ImagingGPUObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->gpu_image->bands);
}

static PyObject *
_gpu_getattr_backend(ImagingGPUObject *self, void *closure) {
    (void)closure;
    if (self->gpu_image->backend_type == GPU_BACKEND_OPENCL)
        return PyUnicode_FromString("OpenCL");
    if (self->gpu_image->backend_type == GPU_BACKEND_CUDA)
        return PyUnicode_FromString("CUDA");
    return PyUnicode_FromString("Unknown");
}

/* -------------------------------------------------------------------- */
/* TYPE DEFINITION                                                       */
/* -------------------------------------------------------------------- */

static struct PyMethodDef _gpu_methods[] = {
    /* Transfer */
    {"to_imaging", (PyCFunction)_gpu_to_imaging, METH_NOARGS},
    {"to_bytes", (PyCFunction)_gpu_to_bytes, METH_NOARGS},

    /* Processing */
    {"gaussian_blur", (PyCFunction)_gpu_gaussian_blur, METH_VARARGS},
    {"box_blur", (PyCFunction)_gpu_box_blur, METH_VARARGS},
    {"unsharp_mask", (PyCFunction)_gpu_unsharp_mask, METH_VARARGS},
    {"filter", (PyCFunction)_gpu_filter, METH_VARARGS},
    {"resize", (PyCFunction)_gpu_resize, METH_VARARGS},
    {"convert", (PyCFunction)_gpu_convert, METH_VARARGS},
    {"transpose", (PyCFunction)_gpu_transpose, METH_VARARGS},
    {"transform", (PyCFunction)_gpu_transform, METH_VARARGS},
    {"chop", (PyCFunction)_gpu_chop, METH_VARARGS},
    {"point_transform", (PyCFunction)_gpu_point_transform, METH_VARARGS},
    {"copy", (PyCFunction)_gpu_copy, METH_NOARGS},

    /* Band operations */
    {"getband", (PyCFunction)_gpu_getband, METH_VARARGS},
    {"putband", (PyCFunction)_gpu_putband, METH_VARARGS},
    {"fillband", (PyCFunction)_gpu_fillband, METH_VARARGS},
    {"split", (PyCFunction)_gpu_split_method, METH_NOARGS},

    /* Point / LUT */
    {"point_lut", (PyCFunction)_gpu_point_lut, METH_VARARGS},

    /* Paste (in-place) */
    {"paste", (PyCFunction)_gpu_paste, METH_VARARGS},

    /* Statistics */
    {"histogram", (PyCFunction)_gpu_histogram, METH_NOARGS},
    {"getbbox", (PyCFunction)_gpu_getbbox, METH_VARARGS},
    {"getextrema", (PyCFunction)_gpu_getextrema, METH_NOARGS},

    /* Effects */
    {"effect_spread", (PyCFunction)_gpu_effect_spread, METH_VARARGS},

    /* Geometry */
    {"crop", (PyCFunction)_gpu_crop, METH_VARARGS},
    {"expand", (PyCFunction)_gpu_expand, METH_VARARGS},
    {"offset", (PyCFunction)_gpu_offset_method, METH_VARARGS},

    /* Image processing */
    {"negative", (PyCFunction)_gpu_negative_method, METH_NOARGS},
    {"posterize", (PyCFunction)_gpu_posterize_method, METH_VARARGS},
    {"solarize", (PyCFunction)_gpu_solarize_method, METH_VARARGS},
    {"equalize", (PyCFunction)_gpu_equalize_method, METH_VARARGS},

    /* Reduce / Rank / Mode filters */
    {"reduce", (PyCFunction)_gpu_reduce_method, METH_VARARGS},
    {"rank_filter", (PyCFunction)_gpu_rank_filter_method, METH_VARARGS},
    {"mode_filter", (PyCFunction)_gpu_mode_filter_method, METH_VARARGS},

    {NULL, NULL} /* sentinel */
};

static PyGetSetDef _gpu_getsetters[] = {
    {"mode", (getter)_gpu_getattr_mode, NULL, "Image mode", NULL},
    {"size", (getter)_gpu_getattr_size, NULL, "Image size (width, height)", NULL},
    {"bands", (getter)_gpu_getattr_bands, NULL, "Number of bands", NULL},
    {"backend", (getter)_gpu_getattr_backend, NULL, "GPU backend name", NULL},
    {NULL}
};

static PyTypeObject ImagingGPU_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ImagingGPU",
    .tp_basicsize = sizeof(ImagingGPUObject),
    .tp_dealloc = (destructor)_gpu_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = _gpu_methods,
    .tp_getset = _gpu_getsetters,
};

/* -------------------------------------------------------------------- */
/* MODULE FUNCTIONS TABLE                                                */
/* -------------------------------------------------------------------- */

static PyMethodDef module_functions[] = {
    /* Backend management */
    {"backend_init",
     (PyCFunction)_gpu_backend_init,
     METH_VARARGS,
     "Initialize the GPU backend. Optional: 1=OpenCL, 2=CUDA, 0=auto."},
    {"backend_shutdown",
     (PyCFunction)_gpu_backend_shutdown,
     METH_NOARGS,
     "Shut down the GPU backend and release resources."},
    {"is_available",
     (PyCFunction)_gpu_is_available,
     METH_NOARGS,
     "Returns True if a GPU backend is active."},
    {"get_backend_name",
     (PyCFunction)_gpu_get_backend_name,
     METH_NOARGS,
     "Returns the name of the active GPU backend."},
    {"get_device_name",
     (PyCFunction)_gpu_get_device_name,
     METH_NOARGS,
     "Returns the name of the GPU device."},

    /* Image creation */
    {"new", (PyCFunction)_gpu_new, METH_VARARGS, "Create a new zeroed GPU image."},
    {"fill",
     (PyCFunction)_gpu_fill,
     METH_VARARGS,
     "Create a new GPU image filled with a color."},
    {"from_imaging",
     (PyCFunction)_gpu_from_imaging,
     METH_VARARGS,
     "Upload a CPU Imaging object to GPU."},
    {"from_bytes",
     (PyCFunction)_gpu_from_bytes,
     METH_VARARGS,
     "Upload raw pixel bytes to GPU. Args: (mode, (w,h), bytes)."},

    /* Compositing (module-level) */
    {"blend", (PyCFunction)_gpu_blend, METH_VARARGS, "Blend two GPU images."},
    {"alpha_composite",
     (PyCFunction)_gpu_alpha_composite,
     METH_VARARGS,
     "Alpha composite two GPU images."},
    {"merge",
     (PyCFunction)_gpu_merge,
     METH_VARARGS,
     "Merge single-band GPU images into a multi-band image."},
    {"linear_gradient",
     (PyCFunction)_gpu_linear_gradient,
     METH_VARARGS,
     "Create a linear gradient GPU image."},
    {"radial_gradient",
     (PyCFunction)_gpu_radial_gradient,
     METH_VARARGS,
     "Create a radial gradient GPU image."},

    {NULL, NULL, 0, NULL}
};

/* -------------------------------------------------------------------- */
/* MODULE INITIALIZATION                                                 */
/* -------------------------------------------------------------------- */

static int
_gpu_module_exec(PyObject *m) {
    if (PyType_Ready(&ImagingGPU_Type) < 0) {
        return -1;
    }
    Py_INCREF(&ImagingGPU_Type);
    if (PyModule_AddObject(m, "ImagingGPU", (PyObject *)&ImagingGPU_Type) < 0) {
        Py_DECREF(&ImagingGPU_Type);
        return -1;
    }

    /* Backend type constants */
    PyModule_AddIntConstant(m, "BACKEND_NONE", GPU_BACKEND_NONE);
    PyModule_AddIntConstant(m, "BACKEND_OPENCL", GPU_BACKEND_OPENCL);
    PyModule_AddIntConstant(m, "BACKEND_CUDA", GPU_BACKEND_CUDA);

    /* Resample filter constants */
    PyModule_AddIntConstant(m, "NEAREST", GPU_RESAMPLE_NEAREST);
    PyModule_AddIntConstant(m, "LANCZOS", GPU_RESAMPLE_LANCZOS);
    PyModule_AddIntConstant(m, "BILINEAR", GPU_RESAMPLE_BILINEAR);
    PyModule_AddIntConstant(m, "BICUBIC", GPU_RESAMPLE_BICUBIC);
    PyModule_AddIntConstant(m, "BOX", GPU_RESAMPLE_BOX);
    PyModule_AddIntConstant(m, "HAMMING", GPU_RESAMPLE_HAMMING);

    /* Transpose operation constants */
    PyModule_AddIntConstant(m, "FLIP_LEFT_RIGHT", GPU_TRANSPOSE_FLIP_LR);
    PyModule_AddIntConstant(m, "FLIP_TOP_BOTTOM", GPU_TRANSPOSE_FLIP_TB);
    PyModule_AddIntConstant(m, "ROTATE_90", GPU_TRANSPOSE_ROTATE_90);
    PyModule_AddIntConstant(m, "ROTATE_180", GPU_TRANSPOSE_ROTATE_180);
    PyModule_AddIntConstant(m, "ROTATE_270", GPU_TRANSPOSE_ROTATE_270);
    PyModule_AddIntConstant(m, "TRANSPOSE", GPU_TRANSPOSE_TRANSPOSE);
    PyModule_AddIntConstant(m, "TRANSVERSE", GPU_TRANSPOSE_TRANSVERSE);

    /* Chop operation constants */
    PyModule_AddIntConstant(m, "CHOP_ADD", GPU_CHOP_ADD);
    PyModule_AddIntConstant(m, "CHOP_SUBTRACT", GPU_CHOP_SUBTRACT);
    PyModule_AddIntConstant(m, "CHOP_MULTIPLY", GPU_CHOP_MULTIPLY);
    PyModule_AddIntConstant(m, "CHOP_SCREEN", GPU_CHOP_SCREEN);
    PyModule_AddIntConstant(m, "CHOP_OVERLAY", GPU_CHOP_OVERLAY);
    PyModule_AddIntConstant(m, "CHOP_DIFFERENCE", GPU_CHOP_DIFFERENCE);
    PyModule_AddIntConstant(m, "CHOP_LIGHTER", GPU_CHOP_LIGHTER);
    PyModule_AddIntConstant(m, "CHOP_DARKER", GPU_CHOP_DARKER);
    PyModule_AddIntConstant(m, "CHOP_ADD_MODULO", GPU_CHOP_ADD_MODULO);
    PyModule_AddIntConstant(m, "CHOP_SUBTRACT_MODULO", GPU_CHOP_SUBTRACT_MODULO);
    PyModule_AddIntConstant(m, "CHOP_SOFT_LIGHT", GPU_CHOP_SOFT_LIGHT);
    PyModule_AddIntConstant(m, "CHOP_HARD_LIGHT", GPU_CHOP_HARD_LIGHT);
    PyModule_AddIntConstant(m, "CHOP_AND", GPU_CHOP_AND);
    PyModule_AddIntConstant(m, "CHOP_OR", GPU_CHOP_OR);
    PyModule_AddIntConstant(m, "CHOP_XOR", GPU_CHOP_XOR);
    PyModule_AddIntConstant(m, "CHOP_INVERT", GPU_CHOP_INVERT);

    /* Availability flags */
    PyObject *have_opencl, *have_cuda;
#ifdef HAVE_OPENCL
    have_opencl = Py_True;
#else
    have_opencl = Py_False;
#endif
#ifdef HAVE_CUDA
    have_cuda = Py_True;
#else
    have_cuda = Py_False;
#endif
    PyModule_AddObjectRef(m, "HAVE_OPENCL", have_opencl);
    PyModule_AddObjectRef(m, "HAVE_CUDA", have_cuda);

    return 0;
}

static PyModuleDef_Slot _gpu_slots[] = {
    {Py_mod_exec, _gpu_module_exec},
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

PyMODINIT_FUNC
PyInit__imaging_gpu(void) {
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        .m_name = "_imaging_gpu",
        .m_doc = "Pillow GPU acceleration module",
        .m_methods = module_functions,
        .m_slots = _gpu_slots,
    };
    return PyModuleDef_Init(&module_def);
}
