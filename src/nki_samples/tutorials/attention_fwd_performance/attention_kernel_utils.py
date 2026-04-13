"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

Utility functions for attention NKI tutorial kernels.
"""
import nki.language as nl
import nki.isa as nisa


def softmax_isa(data, axis=(1,)):
    """Softmax along the given axis using ISA ops.

    Args:
        data: SBUF tensor to compute softmax over
        axis: reduction axis (default: last free dim)

    Returns:
        SBUF tensor with softmax applied
    """
    reduced_shape = list(data.shape)
    for a in axis:
        reduced_shape[a] = 1
    reduced_shape = tuple(reduced_shape)

    row_max = nl.ndarray(reduced_shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, data=data, op=nl.maximum, axis=axis)

    norm = nl.ndarray(data.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=norm, data=data, op0=nl.subtract, operand0=row_max)

    exp_vals = nl.ndarray(data.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_vals, op=nl.exp, data=norm)

    row_sum = nl.ndarray(reduced_shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, data=exp_vals, op=nl.add, axis=axis)

    inv_sum = nl.ndarray(reduced_shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)

    result = nl.ndarray(data.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=result, data=exp_vals, op0=nl.multiply, operand0=inv_sum)

    return result
