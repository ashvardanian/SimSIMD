/**
 *  @brief Scalar types for NumKong low-precision types.
 *  @file python/scalars.c
 *
 *  Pure Python C API implementation of bfloat16, float8_e4m3, and float8_e5m2.
 *  These work without NumPy - they're standalone Python objects.
 *  All arithmetic is performed through float32 using the conversion functions
 *  defined in numkong/types.h.
 *
 *  Usage:
 *    nk.bfloat16(3.14)           → create a bfloat16 scalar
 *    float(nk.bfloat16(3.14))    → convert back to Python float
 *    nk.bfloat16(1.0) + nk.bfloat16(2.0)  → arithmetic
 */

#include "scalars.h"
#include <structmember.h>
#include <stdio.h>

static PyObject *NkBFloat16Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkBFloat16ScalarObject *self = (NkBFloat16ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    if (!PyArg_ParseTuple(args, "|d", &value)) {
        Py_DECREF(self);
        return NULL;
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_bf16(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkBFloat16Scalar_repr(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "bfloat16(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkBFloat16Scalar_str(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkBFloat16Scalar_float(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkBFloat16Scalar_int(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkBFloat16Scalar_hash(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkBFloat16Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    nk_f32_t self_val, other_val;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) {
        nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &self_val);
    }
    else if (PyFloat_Check(self)) { self_val = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { self_val = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) {
        nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &other_val);
    }
    else if (PyFloat_Check(other)) { other_val = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { other_val = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    int result;
    switch (op) {
    case Py_LT: result = self_val < other_val; break;
    case Py_LE: result = self_val <= other_val; break;
    case Py_EQ: result = self_val == other_val; break;
    case Py_NE: result = self_val != other_val; break;
    case Py_GT: result = self_val > other_val; break;
    case Py_GE: result = self_val >= other_val; break;
    default: Py_RETURN_NOTIMPLEMENTED;
    }

    if (result) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

// Arithmetic operations - return bfloat16 result
static PyObject *NkBFloat16Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a + b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a - b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a * b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) { nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "bfloat16 division by zero");
        return NULL;
    }

    result = a / b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_neg(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_pos(NkBFloat16ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkBFloat16Scalar_abs(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkBFloat16Scalar_bool(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkBFloat16Scalar_as_number = {
    .nb_add = NkBFloat16Scalar_add,
    .nb_subtract = NkBFloat16Scalar_sub,
    .nb_multiply = NkBFloat16Scalar_mul,
    .nb_true_divide = NkBFloat16Scalar_div,
    .nb_negative = (unaryfunc)NkBFloat16Scalar_neg,
    .nb_positive = (unaryfunc)NkBFloat16Scalar_pos,
    .nb_absolute = (unaryfunc)NkBFloat16Scalar_abs,
    .nb_bool = (inquiry)NkBFloat16Scalar_bool,
    .nb_float = (unaryfunc)NkBFloat16Scalar_float,
    .nb_int = (unaryfunc)NkBFloat16Scalar_int,
};

PyTypeObject NkBFloat16Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.bfloat16",
    .tp_basicsize = sizeof(NkBFloat16ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkBFloat16Scalar_new,
    .tp_repr = (reprfunc)NkBFloat16Scalar_repr,
    .tp_str = (reprfunc)NkBFloat16Scalar_str,
    .tp_hash = (hashfunc)NkBFloat16Scalar_hash,
    .tp_richcompare = NkBFloat16Scalar_richcompare,
    .tp_as_number = &NkBFloat16Scalar_as_number,
    .tp_doc = "NumKong bfloat16 scalar type (16-bit brain floating point)",
};

static PyObject *NkFloat8E4M3Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat8E4M3ScalarObject *self = (NkFloat8E4M3ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    if (!PyArg_ParseTuple(args, "|d", &value)) {
        Py_DECREF(self);
        return NULL;
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e4m3(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat8E4M3Scalar_repr(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float8_e4m3(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E4M3Scalar_str(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E4M3Scalar_float(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat8E4M3Scalar_int(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat8E4M3Scalar_hash(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat8E4M3Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    nk_f32_t self_val, other_val;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) {
        nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &self_val);
    }
    else if (PyFloat_Check(self)) { self_val = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { self_val = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) {
        nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &other_val);
    }
    else if (PyFloat_Check(other)) { other_val = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { other_val = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    int result;
    switch (op) {
    case Py_LT: result = self_val < other_val; break;
    case Py_LE: result = self_val <= other_val; break;
    case Py_EQ: result = self_val == other_val; break;
    case Py_NE: result = self_val != other_val; break;
    case Py_GT: result = self_val > other_val; break;
    case Py_GE: result = self_val >= other_val; break;
    default: Py_RETURN_NOTIMPLEMENTED;
    }

    if (result) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

// Arithmetic operations for float8_e4m3
static PyObject *NkFloat8E4M3Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a + b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a - b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a * b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) { nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float8_e4m3 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_neg(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_pos(NkFloat8E4M3ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat8E4M3Scalar_abs(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat8E4M3Scalar_bool(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat8E4M3Scalar_as_number = {
    .nb_add = NkFloat8E4M3Scalar_add,
    .nb_subtract = NkFloat8E4M3Scalar_sub,
    .nb_multiply = NkFloat8E4M3Scalar_mul,
    .nb_true_divide = NkFloat8E4M3Scalar_div,
    .nb_negative = (unaryfunc)NkFloat8E4M3Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat8E4M3Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat8E4M3Scalar_abs,
    .nb_bool = (inquiry)NkFloat8E4M3Scalar_bool,
    .nb_float = (unaryfunc)NkFloat8E4M3Scalar_float,
    .nb_int = (unaryfunc)NkFloat8E4M3Scalar_int,
};

PyTypeObject NkFloat8E4M3Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float8_e4m3",
    .tp_basicsize = sizeof(NkFloat8E4M3ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat8E4M3Scalar_new,
    .tp_repr = (reprfunc)NkFloat8E4M3Scalar_repr,
    .tp_str = (reprfunc)NkFloat8E4M3Scalar_str,
    .tp_hash = (hashfunc)NkFloat8E4M3Scalar_hash,
    .tp_richcompare = NkFloat8E4M3Scalar_richcompare,
    .tp_as_number = &NkFloat8E4M3Scalar_as_number,
    .tp_doc = "NumKong float8_e4m3 scalar type (8-bit E4M3 floating point)",
};

static PyObject *NkFloat8E5M2Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat8E5M2ScalarObject *self = (NkFloat8E5M2ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    if (!PyArg_ParseTuple(args, "|d", &value)) {
        Py_DECREF(self);
        return NULL;
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e5m2(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat8E5M2Scalar_repr(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float8_e5m2(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E5M2Scalar_str(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E5M2Scalar_float(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat8E5M2Scalar_int(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat8E5M2Scalar_hash(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat8E5M2Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    nk_f32_t self_val, other_val;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) {
        nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &self_val);
    }
    else if (PyFloat_Check(self)) { self_val = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { self_val = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) {
        nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &other_val);
    }
    else if (PyFloat_Check(other)) { other_val = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { other_val = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    int result;
    switch (op) {
    case Py_LT: result = self_val < other_val; break;
    case Py_LE: result = self_val <= other_val; break;
    case Py_EQ: result = self_val == other_val; break;
    case Py_NE: result = self_val != other_val; break;
    case Py_GT: result = self_val > other_val; break;
    case Py_GE: result = self_val >= other_val; break;
    default: Py_RETURN_NOTIMPLEMENTED;
    }

    if (result) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

// Arithmetic operations for float8_e5m2
static PyObject *NkFloat8E5M2Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a + b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a - b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    result = a * b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a); }
    else if (PyFloat_Check(self)) { a = (nk_f32_t)PyFloat_AS_DOUBLE(self); }
    else if (PyLong_Check(self)) { a = (nk_f32_t)PyLong_AsDouble(self); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) { nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b); }
    else if (PyFloat_Check(other)) { b = (nk_f32_t)PyFloat_AS_DOUBLE(other); }
    else if (PyLong_Check(other)) { b = (nk_f32_t)PyLong_AsDouble(other); }
    else { Py_RETURN_NOTIMPLEMENTED; }

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float8_e5m2 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_neg(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_pos(NkFloat8E5M2ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat8E5M2Scalar_abs(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat8E5M2Scalar_bool(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat8E5M2Scalar_as_number = {
    .nb_add = NkFloat8E5M2Scalar_add,
    .nb_subtract = NkFloat8E5M2Scalar_sub,
    .nb_multiply = NkFloat8E5M2Scalar_mul,
    .nb_true_divide = NkFloat8E5M2Scalar_div,
    .nb_negative = (unaryfunc)NkFloat8E5M2Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat8E5M2Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat8E5M2Scalar_abs,
    .nb_bool = (inquiry)NkFloat8E5M2Scalar_bool,
    .nb_float = (unaryfunc)NkFloat8E5M2Scalar_float,
    .nb_int = (unaryfunc)NkFloat8E5M2Scalar_int,
};

PyTypeObject NkFloat8E5M2Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float8_e5m2",
    .tp_basicsize = sizeof(NkFloat8E5M2ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat8E5M2Scalar_new,
    .tp_repr = (reprfunc)NkFloat8E5M2Scalar_repr,
    .tp_str = (reprfunc)NkFloat8E5M2Scalar_str,
    .tp_hash = (hashfunc)NkFloat8E5M2Scalar_hash,
    .tp_richcompare = NkFloat8E5M2Scalar_richcompare,
    .tp_as_number = &NkFloat8E5M2Scalar_as_number,
    .tp_doc = "NumKong float8_e5m2 scalar type (8-bit E5M2 floating point)",
};

int nk_register_scalar_types(PyObject *module) {
    // Initialize scalar types
    if (PyType_Ready(&NkBFloat16Scalar_Type) < 0) return -1;
    if (PyType_Ready(&NkFloat8E4M3Scalar_Type) < 0) return -1;
    if (PyType_Ready(&NkFloat8E5M2Scalar_Type) < 0) return -1;

    // Add scalar types to module
    Py_INCREF(&NkBFloat16Scalar_Type);
    if (PyModule_AddObject(module, "bfloat16", (PyObject *)&NkBFloat16Scalar_Type) < 0) {
        Py_DECREF(&NkBFloat16Scalar_Type);
        return -1;
    }

    Py_INCREF(&NkFloat8E4M3Scalar_Type);
    if (PyModule_AddObject(module, "float8_e4m3", (PyObject *)&NkFloat8E4M3Scalar_Type) < 0) {
        Py_DECREF(&NkFloat8E4M3Scalar_Type);
        return -1;
    }

    Py_INCREF(&NkFloat8E5M2Scalar_Type);
    if (PyModule_AddObject(module, "float8_e5m2", (PyObject *)&NkFloat8E5M2Scalar_Type) < 0) {
        Py_DECREF(&NkFloat8E5M2Scalar_Type);
        return -1;
    }

    return 0;
}
