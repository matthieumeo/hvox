import numpy as np

from hvox import _nufft

__all__ = ["vis2dirty", "dirty2vis", "compute_psf"]


def vis2dirty(
    uvw_lambda,
    xyz,
    vis,
    wgt_vis=None,
    wgt_dirty=None,
    w_term=True,
    epsilon=1e-3,
    chunked=False,
    max_mem=None,
):
    r"""
    Converts visibilities to a dirty image using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_lambda: np.ndarray((nbaselines, 3))
        UVW coordinates from the measurement set, in wavelengths
    xyz: np.ndarray((nsources, 3))
        Source coordinates from the measurement set
    vis: np.ndarray(nbaselines)
        The input visibilities. Its dtype determines the precision at which computations are done
    wgt_vis: np.ndarray(nbaselines, dtype=vis.dtype), optional
        If present, its values are multiplied to the vis
    wgt_dirty: np.ndarray(nsources, dtype=dirty.dtype), optional
        If present, its values are multiplied to the dirty
    w_term: bool
        It False, drop the 3rd dimension in both domains (i.e., `w` and `z`) for the computation.
    epsilon: float
        Accuracy at which the computation should be done. Must be larger than 1e-9 for double precision
        (`vis` has type np.complex128), and 1e-5 for single precision (`vis` has type np.complex64).
    chunked: bool
        If True, chunking of uvw and sky domains as strategy to subdivide the operation into smaller tasks to
        optimize memory allocation (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further
        information)
    max_mem: int
        (only for chunking strategy) Maximum size of subdivided FFTs
        (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further information)

    Returns
    -------
    np.ndarray(nsources, dtype=float of same precision as `vis`)
        Dirty image

    Notes
    -----
    Nothing for the moment

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       uvw_lambda = rng.random((20, 3)) - 0.5
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = np.ones_like(xx)
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
       visibilities =  rng.random(20).astype("complex")
       visibilities += 1j * rng.random(20)

       dirty = hvox.vis2dirty(
                   uvw_lambda=uvw_lambda,
                   xyz=xyz,
                   vis=visibilities,
               ).reshape(25, 25)
       plt.imshow(dirty)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.title("Dirty image")

    See Also
    --------
    :py:func:`~hvox.core.dirty2vis`, :py:func:`~hvox.core.compute_psf`

    """

    # wgt_vis = flagged_weight
    # wgt_dirty = 1 / jacobian

    # Remove w_term if asked
    uvw_ = uvw_lambda if w_term else uvw_lambda[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply visibility weights
    vis_ = vis * wgt_vis if wgt_vis is not None else vis

    # NUFFT Type 3
    dirty = _nufft.nufft_vis2dirty(xyz_, uvw_, vis_, epsilon, chunked, max_mem)

    # Apply dirty weights
    wgt_dirty = wgt_dirty if wgt_dirty is not None else np.ones_like(dirty)
    wgt_dirty = wgt_dirty / wgt_vis.sum() if wgt_vis is not None else wgt_dirty
    dirty *= wgt_dirty

    return dirty


def dirty2vis(
    uvw_lambda,
    xyz,
    dirty,
    wgt_dirty=None,
    w_term=True,
    epsilon=1e-3,
    chunked=False,
    max_mem=None,
):
    r"""
    Converts a dirty image to visibilities using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_lambda: np.ndarray((nbaselines, 3))
        UVW coordinates from the measurement set, in wavelengths
    xyz: np.ndarray((nsources, 3))
        Source coordinates from the measurement set
    dirty: np.ndarray(nsources)
        The input dirty image. Its dtype determines the precision at which computations are done
    wgt_dirty: np.ndarray(nsources, dtype=dirty.dtype), optional
        If present, its values are multiplied to the dirty
    w_term: bool
        It False, drop the 3rd dimension in both domains (i.e., `w` and `z`) for the computation.
    epsilon: float
        Accuracy at which the computation should be done. Must be larger than 1e-9 for double precision
        (`dirty` has type np.float64), and 1e-5 for single precision (`dirty` has type np.float32).
    chunked: bool
        If True, chunking of uvw and sky domains as strategy to subdivide the operation into smaller tasks to
        optimize memory allocation (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further
        information)
    max_mem: int
        (only for chunking strategy) Maximum size of subdivided FFTs
        (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further information)

    Returns
    -------
    np.ndarray(nbaselines, dtype=float of same precision as `dirty`)
        Visibilities

    Notes
    -----
    Nothing for the moment

     Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       uvw_lambda = rng.random((20, 3)) - 0.5
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = np.ones_like(xx)
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
       dirty =  rng.random(xx.size)

       visibilities = hvox.dirty2vis(
                   uvw_lambda=uvw_lambda,
                   xyz=xyz,
                   dirty=dirty,
               )
       uvw_dist = np.linalg.norm(uvw_lambda, axis=1)
       plt.scatter(uvw_dist, visibilities.real, label="real part")
       plt.scatter(uvw_dist, visibilities.imag, label="imaginary part")
       plt.xlabel("uvw_dist")
       plt.ylabel("visibility")
       plt.legend()

    See Also
    --------
    :py:func:`~hvox.core.vis2dirty`, :py:func:`~hvox.core.compute_psf`

    """

    # wgt_dirty = 1 / jacobian

    nrows, _ = uvw_lambda.shape

    # Remove w_term if asked
    uvw_ = uvw_lambda if w_term else uvw_lambda[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply dirty weights
    dirty_ = dirty * wgt_dirty if wgt_dirty is not None else dirty

    # NUFFT Type 3
    vis = _nufft.nufft_dirty2vis(xyz_, uvw_, dirty_, epsilon, chunked, max_mem)

    return vis


def compute_psf(
    uvw_lambda,
    xyz,
    wgt_vis=None,
    wgt_psf=None,
    w_term=True,
    epsilon=1e-3,
    chunked=False,
    max_mem=None,
):
    r"""
    Computes the point-spread function (PSF) using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_lambda: np.ndarray((nbaselines, 3))
        UVW coordinates from the measurement set, in wavelengths
    xyz: np.ndarray((nsources, 3))
        Source coordinates from the measurement set
    wgt_vis: np.ndarray(nbaselines, dtype=vis.dtype), optional
        If present, its values modulate (multiply) the contribution of each baseline.
    wgt_psf: np.ndarray(nsources, dtype=dirty.dtype), optional
        If present, its values are multiplied to the PSF
    w_term: bool
        It False, drop the 3rd dimension in both domains (i.e., `w` and `z`) for the computation.
    epsilon: float
        Accuracy at which the computation should be done. Must be larger than 1e-9 for double precision
        (`vis` has type np.complex128), and 1e-5 for single precision (`vis` has type np.complex64).
    chunked: bool
        If True, chunking of uvw and sky domains as strategy to subdivide the operation into smaller tasks to
        optimize memory allocation (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further
        information)
    max_mem: int
        (only for chunking strategy) Maximum size of subdivided FFTs
        (see [HVOXpaper]_ and :py:class:`~pycsou.operator.linop.nufft.NUFFT` for further information)

    Returns
    -------
    np.ndarray(nsources, dtype=float of same precision as `vis`)
        PSF image

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       uvw_lambda = rng.random((20, 3)) - 0.5
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = np.ones_like(xx)
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)

       psf = hvox.compute_psf(
                   uvw_lambda=uvw_lambda,
                   xyz=xyz,
               ).reshape(25, 25)
       plt.imshow(psf)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.title("PSF")

    See Also
    --------
    :py:func:`~hvox.core.dirty2vis`, :py:func:`~hvox.core.dirty2vis`

    """
    dtype = np.csingle if np.issubdtype(uvw_lambda.dtype, np.single) else np.cdouble
    vis = np.ones(uvw_lambda.shape[0], dtype=dtype)
    return vis2dirty(
        uvw_lambda,
        xyz,
        vis,
        wgt_vis=wgt_vis,
        wgt_dirty=wgt_psf,
        w_term=w_term,
        epsilon=epsilon,
        chunked=chunked,
        max_mem=max_mem,
    )
