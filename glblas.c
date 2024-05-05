// Copyright (c) 2024 dmaivel

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "glblas.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <epoxy/egl.h>
// #include <EGL/egl.h>

#define FLOATS_PER_PIXEL 4

#define GLBLAS_ASSERT(x, ...) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "GLBLAS_ASSERT: %s:%d:%s: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, #x); \
            fprintf(stderr, " || " __VA_ARGS__); \
            abort(); \
        } \
    } while (0)

#define GLBLAS_ASSERT_STATUS(x, status) \
    if (!(x)) { \
        return status; \
    }

#define IF_NOT_SUCCESS_RETURN(x) \
    status = x; \
    if (status) \
        return status

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef enum _glblas_internal_shader_op {
    OP_GENERIC,

    OP_SSCAL,
    OP_SCOPY,
    OP_SAXPY,
    OP_SDOT,
    OP_SDOTV2_MUL,
    OP_SDOTV2_SUM,
    OP_SASUM,
    OP_SGEMM,
    OP_SGEMM4x4,
    OP_SGEMM4x4_R,

    OP_MAX
} _glblas_internal_shader_op;

typedef struct _glblas_internal_context {
    EGLDisplay dpy;
    EGLint minor, major;

    EGLint n_config;
    EGLConfig config;

    EGLSurface surface;
    EGLContext egl_context;

    int pbuffer_width;
    int pbuffer_height;
    void *pbuffer_host;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
} _glblas_internal_context;

typedef struct _glblas_internal_buffer {
    struct _glblas_internal_buffer *next;

    _glblas_internal_context *context;

    size_t size;
    int width;
    int height;

    bool is_padded;

    unsigned int framebuffer;
    unsigned int texture_colorbuffer;
} _glblas_internal_buffer;

typedef struct _glblas_internal_shader {
    const char * const src;
    unsigned int id;
    unsigned int program;
} _glblas_internal_shader;

static const char *const glblas_vs_src_generic = 
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "   TexCoord = aTexCoord;\n"
    "}";

static const char *const glblas_fs_src_sscal =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform float alpha;\n"
    "uniform int incx;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "   vec4 vx = texture(x, TexCoord);\n"
    "   vx.r = ((index + 0) > max_index || (index + 0) % incx != 0) ? vx.r : vx.r * alpha;\n"
    "   vx.g = ((index + 1) > max_index || (index + 1) % incx != 0) ? vx.g : vx.g * alpha;\n"
    "   vx.b = ((index + 2) > max_index || (index + 2) % incx != 0) ? vx.b : vx.b * alpha;\n"
    "   vx.a = ((index + 3) > max_index || (index + 3) % incx != 0) ? vx.a : vx.a * alpha;\n"
    "   FragColor = vx;\n"
    "}";

static const char *const glblas_fs_src_scopy =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "   vec2 xTexCoord;\n"
    "   vec4 vx = texture(x, TexCoord);\n"
    "   vec4 vy = texture(y, TexCoord);\n"
    "   vy.r = ((index + 0) > max_index) ? vy.r : vx.r;\n"
    "   vy.g = ((index + 1) > max_index) ? vy.g : vx.g;\n"
    "   vy.b = ((index + 2) > max_index) ? vy.b : vx.b;\n"
    "   vy.a = ((index + 3) > max_index) ? vy.a : vx.a;\n"
    "   FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_scopyv2 =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform int incx;\n"
    "uniform int incy;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index && (index + offs) % incy == 0) { \\\n"
    "        int xindex = ((index + offs) + ((index + offs) / incy) * (incx - incy)); \\\n"
    "        xTexCoord.y = float((xindex / 4) / int(dims.x)) / dims.y; \\\n"
    "        xTexCoord.x = float((xindex / 4) % int(dims.x)) / dims.x; \\\n"
    "        vec4 vx = texture(x, xTexCoord); \\\n"
    "        float val; \\\n"
    "        switch (xindex % 4) { \\\n"
    "        case 0: val = vx.r; break; \\\n"
    "        case 1: val = vx.g; break; \\\n"
    "        case 2: val = vx.b; break; \\\n"
    "        case 3: val = vx.a; break; \\\n"
    "        } \\\n"
    "        vy.elem = val; \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    vec4 vy = texture(y, TexCoord);\n"
    "    vec2 xTexCoord;\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_saxpy =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform float alpha;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "   vec2 xTexCoord;\n"
    "   vec4 vx = texture(x, TexCoord);\n"
    "   vec4 vy = texture(y, TexCoord);\n"
    "   vy.r = ((index + 0) > max_index) ? vy.r : vx.r * alpha + vy.r;\n"
    "   vy.g = ((index + 1) > max_index) ? vy.g : vx.g * alpha + vy.g;\n"
    "   vy.b = ((index + 2) > max_index) ? vy.b : vx.b * alpha + vy.b;\n"
    "   vy.a = ((index + 3) > max_index) ? vy.a : vx.a * alpha + vy.a;\n"
    "   FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_saxpyv2 =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform float alpha;\n"
    "uniform int incx;\n"
    "uniform int incy;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index && (index + offs) % incy == 0) { \\\n"
    "        int xindex = ((index + offs) + ((index + offs) / incy) * (incx - incy)); \\\n"
    "        xTexCoord.y = float((xindex / 4) / int(dims.x)) / dims.y; \\\n"
    "        xTexCoord.x = float((xindex / 4) % int(dims.x)) / dims.x; \\\n"
    "        vec4 vx = texture(x, xTexCoord); \\\n"
    "        float val; \\\n"
    "        switch (xindex % 4) { \\\n"
    "        case 0: val = vx.r; break; \\\n"
    "        case 1: val = vx.g; break; \\\n"
    "        case 2: val = vx.b; break; \\\n"
    "        case 3: val = vx.a; break; \\\n"
    "        } \\\n"
    "        vy.elem += val * alpha; \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    vec4 vy = texture(y, TexCoord);\n"
    "    vec2 xTexCoord;\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

// don't use
static const char *const glblas_fs_src_sdot =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform int incx;\n"
    "uniform int incy;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "   if (index != 0) { FragColor = vec4(1, 0, 0, 0); return; }\n"
    "   int ix = 0;\n"
    "   int iy = 0;\n"
    "   vec4 result = vec4(0, 0, 0, 0);\n"
    "   for (int i = 0; i < max_index; i += 4) {\n"
    "       vec2 xTexCoord;\n"
    "       xTexCoord.y = floor((ix) / dims.x) / dims.y;\n"
    "       xTexCoord.x = ((ix) - xTexCoord.y * dims.x * dims.y) / dims.x;\n"
    "       vec2 yTexCoord;\n"
    "       yTexCoord.y = floor((iy) / dims.x) / dims.y;\n"
    "       yTexCoord.x = ((iy) - yTexCoord.y * dims.x * dims.y) / dims.x;\n"
    "       vec4 vx = texture(x, xTexCoord);\n"
    "       vec4 vy = texture(y, yTexCoord);\n"
    "       vec4 vz = vx * vy;\n"
    "       result.r += ((ix + 0) % incx != 0 || (iy + 0) % incy != 0) ? 0 : vz.r;\n"
    "       result.r += ((ix + 1) % incx != 0 || (iy + 1) % incy != 0) ? 0 : vz.g;\n"
    "       result.r += ((ix + 2) % incx != 0 || (iy + 2) % incy != 0) ? 0 : vz.b;\n"
    "       result.r += ((ix + 3) % incx != 0 || (iy + 3) % incy != 0) ? 0 : vz.a;\n"
    "       ix += incx * 4;\n"
    "       iy += incy * 4;\n"
    "   }\n"
    "   FragColor = result;\n"
    "}";

static const char *const glblas_fs_src_sdotv2_mul =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform int incx;\n"
    "uniform int incy;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "   vec2 xTexCoord;\n"
    "   vec4 vx = texture(x, TexCoord);\n"
    "   vec4 vy = texture(y, TexCoord);\n"
    "   vy.r = ((index + 0) > max_index) ? vy.r : vx.r * vy.r;\n"
    "   vy.g = ((index + 1) > max_index) ? vy.g : vx.g * vy.g;\n"
    "   vy.b = ((index + 2) > max_index) ? vy.b : vx.b * vy.b;\n"
    "   vy.a = ((index + 3) > max_index) ? vy.a : vx.a * vy.a;\n"
    "   FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sdotv3_mul =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform int incx;\n"
    "uniform int incy;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index && (index + offs) % incy == 0) { \\\n"
    "        int xindex = ((index + offs) + ((index + offs) / incy) * (incx - incy)); \\\n"
    "        xTexCoord.y = float((xindex / 4) / int(dims.x)) / dims.y; \\\n"
    "        xTexCoord.x = float((xindex / 4) % int(dims.x)) / dims.x; \\\n"
    "        vec4 vx = texture(x, xTexCoord); \\\n"
    "        float val; \\\n"
    "        switch (xindex % 4) { \\\n"
    "        case 0: val = vx.r; break; \\\n"
    "        case 1: val = vx.g; break; \\\n"
    "        case 2: val = vx.b; break; \\\n"
    "        case 3: val = vx.a; break; \\\n"
    "        } \\\n"
    "        vy.elem *= val; \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    vec4 vy = texture(y, TexCoord);\n"
    "    vec2 xTexCoord;\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sdotv2_sum =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform int incx;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5);\n" // we want index of vector, so don't mul by 4
    "   if (max_index == 1) {\n"
    "       vec4 vy = abs(texture(x, TexCoord));\n"
    "       FragColor = index == 0 ? vec4(vy.r + vy.g + vy.b + vy.a, 0, 0, 0) : vec4(1, 0, 0, 0);\n"
    "       return;\n"
    "   }\n"
    "   int halfway = ((max_index / 4));\n" // to-do: document why this is 4 instead of 2?
    "   if (index > halfway) { FragColor = vec4(0, 0, 0, 0); return; }\n"
    "   vec2 xTexCoord;\n"
    "   xTexCoord.y = float(((index + halfway) / incx) / int(dims.x)) / dims.y;\n"
    "   xTexCoord.x = float(((index + halfway) / incx) % int(dims.x)) / dims.x;\n"
    "   vec4 vx = (texture(x, xTexCoord));\n" // CURRENT + HALFWAY
    "   vec4 vy = (texture(x, TexCoord));\n" // CURRENT
    "   vy.r = ((index + 0) > max_index || (index + 0) % incx != 0) ? vy.r : vx.r + vy.r;\n"
    "   vy.g = ((index + 1) > max_index || (index + 1) % incx != 0) ? vy.g : vx.g + vy.g;\n"
    "   vy.b = ((index + 2) > max_index || (index + 2) % incx != 0) ? vy.b : vx.b + vy.b;\n"
    "   vy.a = ((index + 3) > max_index || (index + 3) % incx != 0) ? vy.a : vx.a + vy.a;\n"
    "   FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sasum =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform int incx;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "void main()\n"
    "{\n"
    "   int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5);\n" // we want index of vector, so don't mul by 4
    "   if (max_index == 1) {\n"
    "       vec4 vy = abs(texture(x, TexCoord));\n"
    "       FragColor = index == 0 ? vec4(vy.r + vy.g + vy.b + vy.a, 0, 0, 0) : vec4(1, 0, 0, 0);\n"
    "       return;\n"
    "   }\n"
    "   int halfway = ((max_index / 4));\n" // to-do: document why this is 4 instead of 2?
    "   if (index > halfway) { FragColor = vec4(0, 0, 0, 0); return; }\n"
    "   vec2 xTexCoord;\n"
    "   xTexCoord.y = float(((index + halfway) / incx) / int(dims.x)) / dims.y;\n"
    "   xTexCoord.x = float(((index + halfway) / incx) % int(dims.x)) / dims.x;\n"
    "   vec4 vx = abs(texture(x, xTexCoord));\n" // CURRENT + HALFWAY
    "   vec4 vy = abs(texture(x, TexCoord));\n" // CURRENT
    "   vy.r = ((index + 0) > max_index || (index + 0) % incx != 0) ? vy.r : vx.r + vy.r;\n"
    "   vy.g = ((index + 1) > max_index || (index + 1) % incx != 0) ? vy.g : vx.g + vy.g;\n"
    "   vy.b = ((index + 2) > max_index || (index + 2) % incx != 0) ? vy.b : vx.b + vy.b;\n"
    "   vy.a = ((index + 3) > max_index || (index + 3) % incx != 0) ? vy.a : vx.a + vy.a;\n"
    "   FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sgemm =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D a;\n"
    "uniform sampler2D b;\n"
    "uniform sampler2D c;\n"
    "uniform int lda;\n" // M
    "uniform int ldb;\n" // K
    "uniform int ldc;\n" // M
    "uniform bool aT;\n"
    "uniform bool bT;\n"
    "uniform float alpha;\n"
    "uniform float beta;\n"
    "uniform vec2 dims\n; /* .x = M, .y = N */\n"
    "uniform vec2 adims;\n"
    "uniform vec2 bdims;\n"
    "uniform int m;\n"
    "uniform int n;\n"
    "uniform int k;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index) { \\\n"
    "        float val = 0; \\\n"
    "        int i = (index + offs) % m; \\\n" // row
    "        int j = (index + offs) / m; \\\n" // col
    "        for (int l = 0; l < k; l++) { \\\n"
    "            int aindex = aT ? lda * i + l : lda * l + i; \\\n"
    "            int bindex = bT ? ldb * l + j : ldb * j + l; \\\n"
    "            aTexCoord.y = float((aindex / 4) / ax) / ay; \\\n"
    "            aTexCoord.x = float((aindex / 4) % ax) / ax; \\\n"
    "            bTexCoord.y = float((bindex / 4) / bx) / by; \\\n"
    "            bTexCoord.x = float((bindex / 4) % bx) / bx; \\\n"
    "            vec4 va = texture(a, aTexCoord); \\\n"
    "            vec4 vb = texture(b, bTexCoord); \\\n"
    "            float v0, v1; \\\n"
    "            switch (aindex % 4) { \\\n"
    "            case 0: v0 = va.r; break; \\\n"
    "            case 1: v0 = va.g; break; \\\n"
    "            case 2: v0 = va.b; break; \\\n"
    "            case 3: v0 = va.a; break; \\\n"
    "            } \\\n"
    "            switch (bindex % 4) { \\\n"
    "            case 0: v1 = vb.r; break; \\\n"
    "            case 1: v1 = vb.g; break; \\\n"
    "            case 2: v1 = vb.b; break; \\\n"
    "            case 3: v1 = vb.a; break; \\\n"
    "            } \\\n"
    "            val += v0 * v1; \\\n"
    "        } \\\n"
    "        vy.elem = (alpha * val) + (vy.elem * beta); \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    vec4 vy = (beta != 0.f) ? texture(c, TexCoord) : vec4(0, 0, 0, 0);\n"
    "    vec2 aTexCoord;\n"
    "    vec2 bTexCoord;\n"
    "    int ax = aT ? k : int(adims.x);\n"
    "    int ay = aT ? int(adims.x) : k;\n"
    "    int bx = bT ? int(bdims.y) : k;\n"
    "    int by = bT ? k : int(bdims.y);\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sgemm4x4 =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D a;\n"
    "uniform sampler2D b;\n"
    "uniform sampler2D c;\n"
    "uniform int lda;\n" // M
    "uniform int ldb;\n" // K
    "uniform int ldc;\n" // M
    "uniform bool aT;\n"
    "uniform bool bT;\n"
    "uniform float alpha;\n"
    "uniform float beta;\n"
    "uniform vec2 dims\n; /* .x = M, .y = N */\n"
    "uniform vec2 adims;\n"
    "uniform vec2 bdims;\n"
    "uniform int m;\n"
    "uniform int n;\n"
    "uniform int k;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index) { \\\n"
    "        float val = 0; \\\n"
    "        int i = (index + offs) % m; \\\n" // row
    "        int j = (index + offs) / m; \\\n" // col
    "        for (int l = 0; l < k; l += 4) { \\\n"
    "            int aindex = /*aT ? lda * l + i :*/ lda * i + l; \\\n" // think this is right
    "            int bindex = /*bT ? ldb * l + j :*/ ldb * j + l; \\\n"
    "            aTexCoord.y = float((aindex / 4) / ax) / ay; \\\n"
    "            aTexCoord.x = float((aindex / 4) % ax) / ax; \\\n"
    "            bTexCoord.y = float((bindex / 4) / bx) / by; \\\n"
    "            bTexCoord.x = float((bindex / 4) % bx) / bx; \\\n"
    "            vec4 va = texture(a, aTexCoord); \\\n"
    "            vec4 vb = texture(b, bTexCoord); \\\n"
    "            vb = va * vb; \\\n"
    "            val += vb.r + vb.g + vb.b + vb.a; \\\n"
    "        } \\\n"
    "        vy.elem = (alpha * val) + (vy.elem * beta); \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    vec4 vy = (beta != 0.f) ? texture(c, TexCoord) : vec4(0, 0, 0, 0);\n"
    "    vec2 aTexCoord;\n"
    "    vec2 bTexCoord;\n"
    "    int ax = aT ? k : int(adims.x);\n"
    "    int ay = aT ? int(adims.x) : k;\n"
    "    int bx = bT ? int(bdims.y) : k;\n"
    "    int by = bT ? k : int(bdims.y);\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

static const char *const glblas_fs_src_sgemm4x4_reorder =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D x;\n"
    "uniform sampler2D y;\n"
    "uniform vec2 dims;\n"
    "uniform int max_index;\n"
    "#define kernel(offs, elem) \\\n"
    "    if ((index + offs) < max_index) { \\\n"
    "        int xindex = (fragm + ((max_index / 4) * offs)) /4; \\\n"
    "        xTexCoord.y = float((xindex) / int(dims.x)) / dims.y; \\\n"
    "        xTexCoord.x = float((xindex) % int(dims.x)) / dims.x; \\\n"
    "        vec4 vx = texture(x, xTexCoord); \\\n"
    "        float val; \\\n"
    "        switch (fragm % 4) { \\\n"
    "        case 0: val = vx.r; break; \\\n"
    "        case 1: val = vx.g; break; \\\n"
    "        case 2: val = vx.b; break; \\\n"
    "        case 3: val = vx.a; break; \\\n"
    "        } \\\n"
    "        vy.elem = val; \\\n"
    "    }\n"
    "void main()\n"
    "{\n"
    "    int index = int(gl_FragCoord.y - 0.5) * int(dims.x) + int(gl_FragCoord.x - 0.5) * 4;\n"
    "    int fragm = index / 4;\n"
    "    vec4 vy = vec4(0, 0, 0, 0);\n"
    "    vec2 xTexCoord;\n"
    "    kernel(0, r);\n"
    "    kernel(1, g);\n"
    "    kernel(2, b);\n"
    "    kernel(3, a);\n"
    "    FragColor = vy;\n"
    "}";

_glblas_internal_shader shaders[OP_MAX] = {
    [OP_GENERIC]    = { .src = glblas_vs_src_generic },

    [OP_SSCAL]      = { .src = glblas_fs_src_sscal },
    [OP_SCOPY]      = { .src = glblas_fs_src_scopyv2 },
    [OP_SAXPY]      = { .src = glblas_fs_src_saxpyv2 },
    [OP_SDOT]       = { .src = glblas_fs_src_sdot },
    [OP_SDOTV2_MUL] = { .src = glblas_fs_src_sdotv3_mul },
    [OP_SDOTV2_SUM] = { .src = glblas_fs_src_sdotv2_sum },
    [OP_SASUM]      = { .src = glblas_fs_src_sasum },
    [OP_SGEMM]      = { .src = glblas_fs_src_sgemm },
    [OP_SGEMM4x4]   = { .src = glblas_fs_src_sgemm4x4 },
    [OP_SGEMM4x4_R] = { .src = glblas_fs_src_sgemm4x4_reorder }
};

_glblas_internal_buffer *buffers = NULL;

static const EGLint egl_generic_config[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
};

static void *dynarr_alloc(void **root, size_t next_offset, size_t size)
{
    if (*root == NULL) {
        *root = calloc(1, size);
        return *root;
    }

    void *next;
    for (next = *root; *(void**)(next + next_offset); next = *(void**)(next + next_offset));

    *(void**)(next + next_offset) = calloc(1, size);
    return *(void**)(next + next_offset);
}

static void dynarr_free_element(void **root, size_t next_offset, void *data)
{
    void *prev = NULL;

    for (void *elem = *root; elem;) {
        void *next = *(void**)(elem + next_offset);

        if (elem == data) {
            if (prev == NULL)
                *root = next;
            else
                *(void**)(prev + next_offset) = next;

            free(elem);
        }
        else
            prev = elem;

        elem = next;
    }
}

static void dynarr_free(void **root, size_t next_offset)
{
    if (*(void**)(root)) {
        dynarr_free(*(void**)(root + next_offset), next_offset);
        free(*root);
    }
}

static bool egl_initialize(_glblas_internal_context *context, int pbuffer_width, int pbuffer_height)
{
    EGLint pb_attr[] = {
        EGL_WIDTH, pbuffer_width,
        EGL_HEIGHT, pbuffer_height,
        EGL_NONE,
    };

    context->dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(context->dpy, &context->major, &context->minor))
        return false;

    eglChooseConfig(context->dpy, egl_generic_config, &context->config, 1, &context->n_config);
    context->surface = eglCreatePbufferSurface(context->dpy, context->config, pb_attr);
    eglBindAPI(EGL_OPENGL_API);
    context->egl_context = eglCreateContext(context->dpy, context->config, EGL_NO_CONTEXT, NULL);    
    eglMakeCurrent(context->dpy, context->surface, context->surface, context->egl_context);

    return eglMakeCurrent(context->dpy, context->surface, context->surface, context->egl_context) != EGL_NOT_INITIALIZED;
}

static int check_shader_errors(GLuint shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char* log = (char*)malloc(length);
        glGetShaderInfoLog(shader, length, NULL, log);
        printf("%s", log);
        free(log);
    }

    return status;
}

glblasStatus_t glblasCreate(glblasHandle_t *handle, int width, int height)
{
    _glblas_internal_context *context = calloc(1, sizeof(_glblas_internal_context));

    if (!egl_initialize(context, width, height)) {
        free(context);
        return GLBLAS_STATUS_ALLOC_FAILED;
    }

    // compile shaders
    for (int i = 0; i < OP_MAX; i++) {
        shaders[i].id = glCreateShader(i != OP_GENERIC ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
        glShaderSource(shaders[i].id, 1, &shaders[i].src, NULL);
        glCompileShader(shaders[i].id);

        GLBLAS_ASSERT(check_shader_errors(shaders[i].id), "failed to compile shader %d\n", i);
        // GLBLAS_ASSERT_STATUS(check_shader_errors(shaders[i].id), GLBLAS_STATUS_NOT_SUPPORTED);
    }

    // link shaders
    for (int i = OP_GENERIC + 1; i < OP_MAX; i++) {
        shaders[i].program = glCreateProgram();
        glAttachShader(shaders[i].program, shaders[OP_GENERIC].id);
        glAttachShader(shaders[i].program, shaders[i].id);
        glLinkProgram(shaders[i].program);
        glDeleteShader(shaders[i].id);
    }

    // delete generic
    glDeleteShader(shaders[OP_GENERIC].id);

    float vertices[] = {
        // positions                        // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
    };
    
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &context->VAO);
    glGenBuffers(1, &context->VBO);
    glGenBuffers(1, &context->EBO);

    glBindVertexArray(context->VAO);

    glBindBuffer(GL_ARRAY_BUFFER, context->VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, context->EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    context->pbuffer_host = calloc(width * height * FLOATS_PER_PIXEL, sizeof(float));
    context->pbuffer_width = width;
    context->pbuffer_height = height;

    *handle = context;

    return GLBLAS_STATUS_SUCCESS;
}

void glblasSync()
{
    glFinish();
}

void glblasDestroy(glblasHandle_t ctx)
{
    _glblas_internal_context *context = (_glblas_internal_context*)ctx;

    glDeleteVertexArrays(1, &context->VAO);
    glDeleteBuffers(1, &context->VBO);
    glDeleteBuffers(1, &context->EBO);

    for (_glblas_internal_buffer *buffer = buffers; buffer;) {
        _glblas_internal_buffer *b = buffer;
        buffer = b->next;
        glblasFree(b);
    }

    dynarr_free((void**)&buffers, 0);

    for (int i = 0; i < OP_MAX; i++)
        glDeleteProgram(shaders[i].program);

    free(ctx);
}

static glblasStatus_t get_texture_dimensions(size_t size, int max_width, int max_height, int *out_width, int *out_height, bool *is_padded)
{
    // transform size to contain number of floats
    size_t count = (size + (FLOATS_PER_PIXEL * sizeof(float)) - 1) / (FLOATS_PER_PIXEL * sizeof(float));

    if (is_padded)
        *is_padded = roundf((float)size / (FLOATS_PER_PIXEL * sizeof(float))) != ((float)size / (FLOATS_PER_PIXEL * sizeof(float)));

    // check if buffer is tiny
    if (count <= max_width) {
        *out_width = count;
        *out_height = 1;
        return GLBLAS_STATUS_SUCCESS;
    }

    // otherwise, (max_width, variable height)
    *out_width = max_width;
    *out_height = (count + max_width - 1) / max_width;
    
    return (*out_width <= max_width && *out_height <= max_height) ? GLBLAS_STATUS_SUCCESS : GLBLAS_STATUS_DIMENSION_OVERFLOW;
}

glblasMemory_t glblasMalloc(glblasHandle_t ctx, size_t size)
{
    _glblas_internal_context *context = (_glblas_internal_context*)ctx;
    _glblas_internal_buffer *buf = dynarr_alloc((void**)&buffers, 0, sizeof(_glblas_internal_buffer));

    GLBLAS_ASSERT(size <= context->pbuffer_width * context->pbuffer_height * sizeof(float) * FLOATS_PER_PIXEL, "size (%ld) is out of bounds (max = %ld)\n", size, context->pbuffer_width * context->pbuffer_height * sizeof(float) * FLOATS_PER_PIXEL);

    buf->size = size;
    get_texture_dimensions(size, context->pbuffer_width, context->pbuffer_height, &buf->width, &buf->height, &buf->is_padded);

    glGenFramebuffers(1, &buf->framebuffer);
    glGenTextures(1, &buf->texture_colorbuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, buf->framebuffer);
    glBindTexture(GL_TEXTURE_2D, buf->texture_colorbuffer);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, buf->width, buf->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // GL_LINEAR changes the values, so use nearest
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf->texture_colorbuffer, 0);

    buf->context = context;

    return (glblasMemory_t)buf;
}

static inline _glblas_internal_buffer *get_buffer_from_address(size_t addr)
{
    for (_glblas_internal_buffer *buf = buffers; buf; buf = buf->next) {
        if (addr == (size_t)buf)
            return buf;
    }
    return NULL;
}

glblasStatus_t glblasMemcpy(void *dst, void *src, size_t size, glblasMemcpyKind_t kind)
{
    _glblas_internal_buffer *buf_dst = get_buffer_from_address((size_t)dst);
    _glblas_internal_buffer *buf_src = get_buffer_from_address((size_t)src);

    _glblas_internal_buffer *buf;

    if (kind == glblasMemcpyInfer) {
        // GLBLAS_ASSERT(buf_dst != NULL || buf_src != NULL, "invalid addresses (%p) (%p)\n", buf_dst, buf_src);
        GLBLAS_ASSERT_STATUS(buf_dst != NULL || buf_src != NULL, GLBLAS_STATUS_INVALID_VALUE);

        if (buf_dst && buf_src)
            kind = glblasMemcpyDeviceToDevice;
        else if (buf_dst == NULL)
            kind = glblasMemcpyDeviceToHost;
        else
            kind = glblasMemcpyHostToDevice;
    }

    switch (kind) {
    case glblasMemcpyHostToDevice:
        buf = buf_dst;
        break;
    case glblasMemcpyDeviceToHost:
        buf = buf_src;
        break;
    case glblasMemcpyDeviceToDevice:
        // GLBLAS_ASSERT(false, "prefer glblasScopy over glblasMemcpyDeviceToDevice\n");
        // buf = buf_dst;
        // break;
        return GLBLAS_STATUS_NOT_SUPPORTED;
    default:
        break;
    }

    // GLBLAS_ASSERT(buf, "invalid buffer\n");
    // GLBLAS_ASSERT(size <= buf->size, "size (%ld) is likely too large, buffer size is %ld\n", size, buf->size);

    GLBLAS_ASSERT_STATUS(buf && size <= buf->size, GLBLAS_STATUS_INVALID_VALUE);

    _glblas_internal_context *context = buf->context;

    switch (kind) {
    case glblasMemcpyHostToDevice:
        if (buf->is_padded || buf->size != size) {
            memcpy(context->pbuffer_host, src, size);
            glBindTexture(GL_TEXTURE_2D, buf->texture_colorbuffer);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf->width, buf->height, GL_RGBA, GL_FLOAT, context->pbuffer_host);
        }
        else {
            glBindTexture(GL_TEXTURE_2D, buf->texture_colorbuffer);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, buf->width, buf->height, GL_RGBA, GL_FLOAT, src);
        }
        break;

    case glblasMemcpyDeviceToHost:
        if (buf->is_padded || buf->size != size) {
            glBindFramebuffer(GL_FRAMEBUFFER, buf->framebuffer);
            glReadPixels(0, 0, buf->width, buf->height, GL_RGBA, GL_FLOAT, context->pbuffer_host);
            memcpy(dst, context->pbuffer_host, size);
        }
        else {
            glBindFramebuffer(GL_FRAMEBUFFER, buf->framebuffer);
            glReadPixels(0, 0, buf->width, buf->height, GL_RGBA, GL_FLOAT, dst);
        }
        break;

    case glblasMemcpyDeviceToDevice:
        glblasMemcpy(context->pbuffer_host, src, buf->size, glblasMemcpyDeviceToHost);
        glblasMemcpy(dst, context->pbuffer_host, buf->size, glblasMemcpyHostToDevice);
        break;

    default:
        break;
    }

    return GLBLAS_STATUS_SUCCESS;
}

void glblasFree(glblasMemory_t buf)
{
    _glblas_internal_buffer *buffer = (_glblas_internal_buffer*)buf;
    
    glDeleteFramebuffers(1, &buffer->framebuffer);
    glDeleteTextures(1, &buffer->texture_colorbuffer);

    dynarr_free_element((void**)&buffers, 0, buf);
}

// swap x & y
glblasStatus_t glblasSswap(int N, glblasMemory_t x, int incx, glblasMemory_t y, int incy)
{
    // infer context from x
    glblasMemory_t temp = glblasMalloc(((_glblas_internal_buffer*)x)->context, N * sizeof(float));
    glblasScopy(N, x, 1, temp, 1);

    glblasScopy(N, y, incy, x, incx); // copy y into x
    glblasScopy(N, temp, incx, y, incy); // copy x into y

    glblasFree(temp);

    return GLBLAS_STATUS_SUCCESS;
}

static glblasStatus_t get_op_dims(const size_t N, _glblas_internal_buffer *dev, _glblas_internal_context *context, int *width, int *height)
{
    if (N == dev->size / sizeof(float)) {
        *width = dev->width;
        *height = dev->height;
    }
    else {
        get_texture_dimensions(N * sizeof(float), context->pbuffer_width, context->pbuffer_height, width, height, NULL);
    }

    return GLBLAS_STATUS_SUCCESS;
}

// x = a*x
glblasStatus_t glblasSscal(int N, const float alpha, glblasMemory_t x, int incx)
{
    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_context *context = device_x->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(N, device_x, context, &width, &height));

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SSCAL].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SSCAL].program, "x"), 0);

    glUniform1f(glGetUniformLocation(shaders[OP_SSCAL].program, "alpha"), alpha);

    glUniform2f(glGetUniformLocation(shaders[OP_SSCAL].program, "dims"), width, height);
    glUniform1i(glGetUniformLocation(shaders[OP_SSCAL].program, "max_index"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SSCAL].program, "incx"), incx);

    glBindFramebuffer(GL_FRAMEBUFFER, device_x->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    return GLBLAS_STATUS_SUCCESS;
}

// copy x into y
glblasStatus_t glblasScopy(int N, const glblasMemory_t x, int incx, glblasMemory_t y, int incy)
{
    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
    _glblas_internal_context *context = device_x->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(N, device_y, context, &width, &height));

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SCOPY].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SCOPY].program, "x"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, device_y->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SCOPY].program, "y"), 1);

    glUniform2f(glGetUniformLocation(shaders[OP_SCOPY].program, "dims"), width, height);
    glUniform1i(glGetUniformLocation(shaders[OP_SCOPY].program, "max_index"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SCOPY].program, "incx"), incx);
    glUniform1i(glGetUniformLocation(shaders[OP_SCOPY].program, "incy"), incy);

    glBindFramebuffer(GL_FRAMEBUFFER, device_y->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    return GLBLAS_STATUS_SUCCESS;
}

// y = a*x + y
glblasStatus_t glblasSaxpy(int N, const float alpha, const glblasMemory_t x, int incx, glblasMemory_t y, int incy)
{
    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
    _glblas_internal_context *context = device_x->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(N, device_y, context, &width, &height));

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SAXPY].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SAXPY].program, "x"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, device_y->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SAXPY].program, "y"), 1);

    glUniform1f(glGetUniformLocation(shaders[OP_SAXPY].program, "alpha"), alpha);

    glUniform2f(glGetUniformLocation(shaders[OP_SAXPY].program, "dims"), width, height);
    glUniform1i(glGetUniformLocation(shaders[OP_SAXPY].program, "max_index"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SAXPY].program, "incx"), incx);
    glUniform1i(glGetUniformLocation(shaders[OP_SAXPY].program, "incy"), incy);

    glBindFramebuffer(GL_FRAMEBUFFER, device_y->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    return GLBLAS_STATUS_SUCCESS;
}

static void glblas_sdotv2_mul(int N, const glblasMemory_t x, int incx, glblasMemory_t y, int incy)
{
    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
    _glblas_internal_context *context = device_x->context;

    int width, height;
    get_op_dims(N, device_y, context, &width, &height);

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SDOTV2_MUL].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "x"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, device_y->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "y"), 1);

    glUniform2f(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "dims"), width, height);
    glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "max_index"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "incx"), incx);
    glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_MUL].program, "incy"), incy);

    glBindFramebuffer(GL_FRAMEBUFFER, device_y->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

static void glblas_sdotv2_sum(int N, glblasMemory_t result, const glblasMemory_t x, int incx)
{
    //GLBLAS_ASSERT(N % FLOATS_PER_PIXEL == 0, "N (%d) must be divisible by 2\n", N); // to-do: maybe support?

    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_result = (_glblas_internal_buffer*)result;
    _glblas_internal_context *context = device_x->context;

    int width, height;
    get_op_dims(N, device_x, context, &width, &height);

    _glblas_internal_buffer *temp = glblasMalloc(device_x->context, N * sizeof(float));
    glblasScopy(N, x, 1, temp, 1);

    glblasSync();

    for (int tN = N / 2; tN != 0; tN /= 2) {
        if (tN < 4)
            tN = 1;

        glViewport(0, 0, width, height);
        glUseProgram(shaders[OP_SDOTV2_SUM].program);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, temp->texture_colorbuffer);
        glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_SUM].program, "x"), 0);

        glUniform2f(glGetUniformLocation(shaders[OP_SDOTV2_SUM].program, "dims"), width, height);
        glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_SUM].program, "max_index"), tN);
        glUniform1i(glGetUniformLocation(shaders[OP_SDOTV2_SUM].program, "incx"), incx);

        glBindFramebuffer(GL_FRAMEBUFFER, temp->framebuffer);
        glBindVertexArray(context->VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glblasSync();
    }

    glblasScopy(1, temp, 1, result, 1);

    glblasFree(temp);
}

// dot product
glblasStatus_t glblasSdot(int N, glblasMemory_t result, const glblasMemory_t x, int incx, const glblasMemory_t y, int incy)
{
    _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
    _glblas_internal_buffer *savedy = glblasMalloc(device_y->context, N * sizeof(float));
    glblasStatus_t status;

    IF_NOT_SUCCESS_RETURN(glblasScopy(N, y, 1, savedy, 1));

    glblasSync();

    glblas_sdotv2_mul(N, x, incx, savedy, incy);
    glblas_sdotv2_sum(N, result, savedy, 1);

    glblasFree(savedy);

    return GLBLAS_STATUS_SUCCESS;
}

// dot product
// void glblasSdot(int N, glblasMemory_t result, const glblasMemory_t x, int incx, const glblasMemory_t y, int incy)
// {
//     _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
//     _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
//     _glblas_internal_buffer *device_result = (_glblas_internal_buffer*)result;
//     _glblas_internal_context *context = device_x->context;

//     int width, height;
//     if (N == device_y->size / sizeof(float)) {
//         width = device_y->width;
//         height = device_y->height;
//     }
//     else {
//         get_texture_dimensions(N * sizeof(float), context->pbuffer_width, context->pbuffer_height, &width, &height, NULL);
//     }

//     glViewport(0, 0, width, height);
//     glUseProgram(shaders[OP_SDOT].program);

//     glActiveTexture(GL_TEXTURE0);
//     glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
//     glUniform1i(glGetUniformLocation(shaders[OP_SDOT].program, "x"), 0);
//     glActiveTexture(GL_TEXTURE1);
//     glBindTexture(GL_TEXTURE_2D, device_y->texture_colorbuffer);
//     glUniform1i(glGetUniformLocation(shaders[OP_SDOT].program, "y"), 1);

//     glUniform2f(glGetUniformLocation(shaders[OP_SDOT].program, "dims"), width, height);
//     glUniform1i(glGetUniformLocation(shaders[OP_SDOT].program, "max_index"), N);
//     glUniform1i(glGetUniformLocation(shaders[OP_SDOT].program, "incx"), incx);
//     glUniform1i(glGetUniformLocation(shaders[OP_SDOT].program, "incy"), incy);

//     glBindFramebuffer(GL_FRAMEBUFFER, device_result->framebuffer);
//     glActiveTexture(GL_TEXTURE2);
//     glBindTexture(GL_TEXTURE_2D, device_result->texture_colorbuffer);
//     glBindVertexArray(context->VAO);
//     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
// }

// sum of abs values
glblasStatus_t glblasSasum(int N, glblasMemory_t result, const glblasMemory_t x, int incx)
{
    // GLBLAS_ASSERT(N % FLOATS_PER_PIXEL == 0, "N (%d) must be divisible by 2\n", N); // to-do: maybe support?

    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_result = (_glblas_internal_buffer*)result;
    _glblas_internal_context *context = device_x->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(N, device_x, context, &width, &height));

    _glblas_internal_buffer *temp = glblasMalloc(device_x->context, N * sizeof(float));
    glblasScopy(N, x, 1, temp, 1);

    glblasSync();

    for (int tN = N / 2; tN != 0; tN /= 2) {
        if (tN < 4)
            tN = 1;

        glViewport(0, 0, width, height);
        glUseProgram(shaders[OP_SASUM].program);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, temp->texture_colorbuffer);
        glUniform1i(glGetUniformLocation(shaders[OP_SASUM].program, "x"), 0);

        glUniform2f(glGetUniformLocation(shaders[OP_SASUM].program, "dims"), width, height);
        glUniform1i(glGetUniformLocation(shaders[OP_SASUM].program, "max_index"), tN);
        glUniform1i(glGetUniformLocation(shaders[OP_SASUM].program, "incx"), incx);

        glBindFramebuffer(GL_FRAMEBUFFER, temp->framebuffer);
        glBindVertexArray(context->VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glblasSync();
    }

    glblasScopy(1, temp, 1, result, 1);
    glblasFree(temp);

    return GLBLAS_STATUS_SUCCESS;
}

// matrix matrix multiply
glblasStatus_t glblasSgemm( glblasOperation_t transa, glblasOperation_t transb
                          , int M, int N, int K, const float alpha
                          , const glblasMemory_t a, const int lda
                          , const glblasMemory_t b, const int ldb, const float beta
                          , glblasMemory_t c, const int ldc )
{
    // GLBLAS_ASSERT(M >= 0 && N >= 0 && K >= 0, "M, N, K must be 0 or positive\n"); // lol
    // GLBLAS_ASSERT(lda >= MAX(1, transa ? K : M), "lda out of range\n");
    // GLBLAS_ASSERT(ldb >= MAX(1, transb ? N : K), "ldb out of range\n");
    // GLBLAS_ASSERT(ldc >= MAX(1, M), "ldc out of range\n");

    GLBLAS_ASSERT_STATUS(M >= 0 && N >= 0 && K >= 0, GLBLAS_STATUS_INVALID_VALUE);
    GLBLAS_ASSERT_STATUS(lda >= MAX(1, transa ? K : M) || ldb >= MAX(1, transb ? N : K) || ldc >= MAX(1, M), GLBLAS_STATUS_DIMENSION_OVERFLOW);

    _glblas_internal_buffer *device_a = (_glblas_internal_buffer*)a;
    _glblas_internal_buffer *device_b = (_glblas_internal_buffer*)b;
    _glblas_internal_buffer *device_c = (_glblas_internal_buffer*)c;
    _glblas_internal_context *context = device_c->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(M * N, device_c, context, &width, &height));

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SGEMM].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_a->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "a"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, device_b->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "b"), 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, device_c->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "c"), 2);

    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM].program, "dims"), width, height);
    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM].program, "adims"), device_a->width, device_a->height);
    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM].program, "bdims"), device_b->width, device_b->height);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "max_index"), M * N);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "m"), M);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "n"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "k"), K);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "lda"), lda);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "ldb"), ldb);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "ldc"), ldc);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "aT"), transa);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM].program, "bT"), transb);
    glUniform1f(glGetUniformLocation(shaders[OP_SGEMM].program, "alpha"), alpha);
    glUniform1f(glGetUniformLocation(shaders[OP_SGEMM].program, "beta"), beta);

    glBindFramebuffer(GL_FRAMEBUFFER, device_c->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    return GLBLAS_STATUS_SUCCESS;
}

static void glblas_sgemm4x4_reorder(int N, const glblasMemory_t x, glblasMemory_t y)
{
    _glblas_internal_buffer *device_x = (_glblas_internal_buffer*)x;
    _glblas_internal_buffer *device_y = (_glblas_internal_buffer*)y;
    _glblas_internal_context *context = device_x->context;

    int width, height;
    get_op_dims(N, device_y, context, &width, &height);

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SGEMM4x4_R].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device_x->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4_R].program, "x"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, device_y->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4_R].program, "y"), 1);

    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM4x4_R].program, "dims"), width, height);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4_R].program, "max_index"), N);

    glBindFramebuffer(GL_FRAMEBUFFER, device_y->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

glblasStatus_t glblasSgemm4x4( glblasOperation_t transa, glblasOperation_t transb
                             , int M, int N, int K, const float alpha
                             , const glblasMemory_t a, const int lda
                             , const glblasMemory_t b, const int ldb, const float beta
                             , glblasMemory_t c, const int ldc )
{
    // GLBLAS_ASSERT(M % 4 == 0 && N % 4 == 0 && K % 4 == 0 && M == N && M == K, "M, N, K must be multiples of 4 and form a square\n");
    // GLBLAS_ASSERT(M >= 0 && N >= 0 && K >= 0, "M, N, K must be 0 or positive\n"); // lol
    // GLBLAS_ASSERT(lda >= MAX(1, transa ? K : M), "lda out of range\n");
    // GLBLAS_ASSERT(ldb >= MAX(1, transb ? N : K), "ldb out of range\n");
    // GLBLAS_ASSERT(ldc >= MAX(1, M), "ldc out of range\n");
    
    GLBLAS_ASSERT_STATUS(M % 4 == 0 && N % 4 == 0 && K % 4 == 0 && M == N && M == K && M >= 0 && N >= 0 && K >= 0, GLBLAS_STATUS_INVALID_VALUE);
    GLBLAS_ASSERT_STATUS(lda >= MAX(1, transa ? K : M) || ldb >= MAX(1, transb ? N : K) || ldc >= MAX(1, M), GLBLAS_STATUS_DIMENSION_OVERFLOW);

    _glblas_internal_buffer *device_a = (_glblas_internal_buffer*)a;
    _glblas_internal_buffer *device_b = (_glblas_internal_buffer*)b;
    _glblas_internal_buffer *device_c = (_glblas_internal_buffer*)c;
    _glblas_internal_context *context = device_c->context;
    glblasStatus_t status;

    int width, height;
    IF_NOT_SUCCESS_RETURN(get_op_dims(M * N, device_c, context, &width, &height));

    _glblas_internal_buffer *reordered_a = transa ? NULL : glblasMalloc(context, M * K * sizeof(float));
    _glblas_internal_buffer *reordered_b = transb ? glblasMalloc(context, K * N * sizeof(float)) : NULL;

    if (!transa)
        glblas_sgemm4x4_reorder(M * K, device_a, reordered_a);
    if (transb)
        glblas_sgemm4x4_reorder(K * N, device_b, reordered_b);

    _glblas_internal_buffer *u_a = transa ? device_a : reordered_a;
    _glblas_internal_buffer *u_b = transb ? reordered_b : device_b;

    glViewport(0, 0, width, height);
    glUseProgram(shaders[OP_SGEMM4x4].program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, u_a->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "a"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, u_b->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "b"), 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, device_c->texture_colorbuffer);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "c"), 2);

    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "dims"), width, height);
    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "adims"), u_a->width, u_a->height);
    glUniform2f(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "bdims"), u_b->width, u_b->height);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "max_index"), M * N);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "m"), M);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "n"), N);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "k"), K);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "lda"), lda);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "ldb"), ldb);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "ldc"), ldc);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "aT"), transa);
    glUniform1i(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "bT"), transb);
    glUniform1f(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "alpha"), alpha);
    glUniform1f(glGetUniformLocation(shaders[OP_SGEMM4x4].program, "beta"), beta);

    glBindFramebuffer(GL_FRAMEBUFFER, device_c->framebuffer);
    glBindVertexArray(context->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    if (reordered_a != NULL)
        glblasFree(reordered_a);
    if (reordered_b != NULL)
        glblasFree(reordered_b);

    return GLBLAS_STATUS_SUCCESS;
}