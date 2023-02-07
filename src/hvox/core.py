import numpy as np

from hvox import _nufft

__all__ = ["vis2dirty", "dirty2vis", "compute_psf"]


def vis2dirty(
    uvw,
    xyz,
    vis,
    mesh="dcos",
    wgt=None,
    normalisation="xyz",
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
    uvw: np.ndarray((nbaselines, 3))
        UVW coordinates from the measurement set, in wavelengths
    xyz: np.ndarray((nsources, 3))
        Source coordinates from the measurement set
    vis: np.ndarray(nbaselines)
        The input visibilities. Its dtype determines the precision at which computations are done
    wgt_vis: np.ndarray(nbaselines, dtype=vis.dtype)
        If present, its values are multiplied to the vis
    wgt_dirty: np.ndarray(nsources, dtype=dirty.dtype)
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
       uvw_lambda = np.concatenate((uvw_lambda, -uvw_lambda), 0)
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

    # Remove w_term if asked
    uvw_ = uvw if w_term else uvw[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply visibility weights (flagged weights)
    vis_ = vis * wgt if wgt is not None else vis

    # NUFFT Type 3
    dirty = _nufft.nufft_vis2dirty(xyz_, uvw_, vis_, epsilon, chunked, max_mem)

    # Apply dirty weights
    if mesh == "dcos":
        jacobian = xyz[:, -1] + 1.0
        dirty /= jacobian

    if normalisation is not None:
        if normalisation in ["uvw", "both"]:
            dirty /= wgt.sum() if wgt is not None else len(uvw)
        if normalisation in ["xyz", "both"]:
            dirty /= len(xyz)

    return dirty


def dirty2vis(
    uvw,
    xyz,
    dirty,
    mesh="dcos",
    wgt=None,
    normalisation="xyz",
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
    uvw: np.ndarray((nbaselines, 3))
        UVW coordinates from the measurement set, in wavelengths
    xyz: np.ndarray((nsources, 3))
        Source coordinates from the measurement set
    dirty: np.ndarray(nsources)
        The input dirty image. Its dtype determines the precision at which computations are done
    wgt_vis: np.ndarray(nbaselines, dtype=vis.dtype), optional
        If present, its values are multiplied to the vis
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
       uvw_lambda = np.concatenate((uvw_lambda, -uvw_lambda), 0)
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

    nrows, _ = uvw.shape

    # Remove w_term if asked
    uvw_ = uvw if w_term else uvw[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply dirty weights
    if mesh == "dcos":
        jacobian = xyz[:, -1] + 1.0
        dirty /= jacobian

    # NUFFT Type 3
    vis = _nufft.nufft_dirty2vis(xyz_, uvw_, dirty, epsilon, chunked, max_mem)

    # Apply visibility weights
    vis_ = vis * wgt if wgt is not None else vis

    if normalisation is not None:
        if normalisation in ["uvw", "both"]:
            vis_ /= wgt.sum() if wgt is not None else len(uvw)
        if normalisation in ["xyz", "both"]:
            vis_ /= len(xyz)

    return vis_


def compute_psf(
    uvw,
    xyz,
    xyz_center=np.array([0.0, 0.0, 0.0]),
    mesh="dcos",
    wgt=None,
    normalisation="both",
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
    uvw: np.ndarray((nbaselines, 3))
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
       uvw_lambda = np.concatenate((uvw_lambda, -uvw_lambda), 0)
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

    dtype = np.csingle if np.issubdtype(uvw.dtype, np.single) else np.cdouble

    phase_center = np.exp(-1j * 2 * np.pi * (uvw.dot(xyz_center))).astype(dtype)

    if mesh == "dcos":
        # Direction cosines:
        # PSF = (1 / (Npix**2)) * (1/jacobian) * (1/jacobian[xyz0]) * NUFFT_adjoint * phase(xyz0)
        jacobian = xyz_center[..., -1] + 1.0
        phase_center /= jacobian

    return vis2dirty(
        uvw,
        xyz,
        phase_center,
        mesh=mesh,
        wgt=wgt,
        normalisation=normalisation,
        w_term=w_term,
        epsilon=epsilon,
        chunked=chunked,
        max_mem=max_mem,
    )
