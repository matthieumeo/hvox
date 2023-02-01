import multiprocessing
import warnings

import numpy as np
import pycsou.operator.linop as pycl
import pycsou.util as pycu


def nufft_builder(xyz, vlambda, epsilon, chunked, max_mem):
    kwargs = {}
    if epsilon is None:
        epsilon = 1e-6
        warnings.warn("Epsilon is not defined, using NUFFT with default `eps=1e-6`.")

    if chunked:
        max_mem = max_mem if max_mem is not None else 500
        kwargs.update(
            {
                "chunked": True,
                "parallel": True,
                "nthreads": multiprocessing.cpu_count() // 4,
            }
        )
    else:
        kwargs.update({"nthreads": multiprocessing.cpu_count()})

    nufft = pycl.NUFFT.type3(
        x=xyz, z=2 * np.pi * vlambda, real=True, isign=-1, eps=epsilon, **kwargs
    )

    if chunked:
        # auto-determine a good x/z chunking strategy
        nufft.allocate(*nufft.auto_chunk(max_mem=max_mem), enable_warnings=False)
        xyz_idx, xyz_chunks = nufft.order("x")  # get a good x-ordering
        uvw_lambda_idx, uvw_lambda_chunks = nufft.order("z")  # get a good z-ordering
        nufft = pycl.NUFFT.type3(
            x=xyz[xyz_idx],
            z=2 * np.pi * vlambda[uvw_lambda_idx],
            real=True,
            isign=-1,
            eps=epsilon,
            **kwargs
        )

        nufft.allocate(xyz_chunks, uvw_lambda_chunks, direct_eval_threshold=1e4)

        if isinstance(xyz_idx, slice):
            xyz_idx = np.arange(xyz_idx.stop)
        if isinstance(uvw_lambda_idx, slice):
            uvw_lambda_idx = np.arange(uvw_lambda_idx.stop - 1)

        sort_func = {
            "dc": {
                "fw": lambda _: _[xyz_idx],
                "bw": lambda _: _[np.argsort(xyz_idx)],
            },
            "vl": {
                "fw": lambda _: _[uvw_lambda_idx],
                "bw": lambda _: _[np.argsort(uvw_lambda_idx)],
            },
        }

    else:
        identity = lambda _: _
        sort_func = {
            "dc": {
                "fw": identity,
                "bw": identity,
            },
            "vl": {
                "fw": identity,
                "bw": identity,
            },
        }

    return nufft, sort_func


def nufft_vis2dirty(
    xyz,
    uvw_lambda,
    visibilities,
    epsilon,
    chunked,
    max_mem,
):
    nufft, sort_func = nufft_builder(xyz, uvw_lambda, epsilon, chunked, max_mem)
    input_data = sort_func["vl"]["fw"](visibilities)
    input_data = pycu.view_as_real(input_data)
    dirty = nufft.adjoint(input_data)
    dirty = sort_func["dc"]["bw"](dirty)
    return dirty


def nufft_dirty2vis(
    xyz,
    uvw_lambda,
    dirty,
    epsilon,
    chunked,
    max_mem,
):
    nufft, sort_func = nufft_builder(xyz, uvw_lambda, epsilon, chunked, max_mem)

    input_data = sort_func["dc"]["fw"](dirty)
    visibilities = pycu.view_as_complex(nufft(input_data))
    visibilities = sort_func["vl"]["bw"](visibilities)
    return visibilities
