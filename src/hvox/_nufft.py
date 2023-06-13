import multiprocessing
import warnings

import numpy as np
import pycsou.operator.linop as pycl
import pycsou.util as pycu


def nufft_builder(xyz, vlambda, real, epsilon, nufft_kwargs):
    # Get NUFFT args
    max_mem = nufft_kwargs.get("max_mem", 512)
    nufft_kwargs = dict(
        enable_warnings=nufft_kwargs.get("enable_warnings", True),
        chunked=nufft_kwargs.get("chunked", True),
        parallel=nufft_kwargs.get("parallel", True),
        nthreads=nufft_kwargs.get("nthreads", None),
    )
    if nufft_kwargs["nthreads"] is None:
        div = 4 if nufft_kwargs["chunked"] else 2
        nufft_kwargs.update({"nthreads": multiprocessing.cpu_count() // div})

    nufft = pycl.NUFFT.type3(
        x=xyz, z=2 * np.pi * vlambda, real=real, isign=-1, eps=epsilon, **nufft_kwargs
    )

    if nufft_kwargs["chunked"]:
        # auto-determine a good x/z chunking strategy
        nufft.allocate(*nufft.auto_chunk(max_mem=max_mem))
        xyz_idx, xyz_chunks = nufft.order("x")  # get a good x-ordering
        uvw_lambda_idx, uvw_lambda_chunks = nufft.order("z")  # get a good z-ordering
        nufft = pycl.NUFFT.type3(
            x=xyz[xyz_idx],
            z=2 * np.pi * vlambda[uvw_lambda_idx],
            real=real,
            isign=-1,
            eps=epsilon,
            **nufft_kwargs
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
    real,
    epsilon,
    nufft_kwargs
):
    nufft, sort_func = nufft_builder(xyz, uvw_lambda, real, epsilon, nufft_kwargs)
    input_data = sort_func["vl"]["fw"](visibilities)
    input_data = pycu.view_as_real(input_data)
    dirty = nufft.adjoint(input_data)
    dirty = sort_func["dc"]["bw"](dirty)
    return dirty


def nufft_dirty2vis(
    xyz,
    uvw_lambda,
    dirty,
    real,
    epsilon,
    nufft_kwargs
):
    nufft, sort_func = nufft_builder(xyz, uvw_lambda, real, epsilon, nufft_kwargs)

    input_data = sort_func["dc"]["fw"](dirty)
    visibilities = pycu.view_as_complex(nufft(input_data))
    visibilities = sort_func["vl"]["bw"](visibilities)
    return visibilities
